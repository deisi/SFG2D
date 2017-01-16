"""Module for Widget classes."""

import warnings
import os
from glob import glob
import numpy as np
from scipy.signal import medfilt2d, medfilt
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

from .io.veronica import read_auto
from .io.allYouCanEat import AllYouCanEat, x_pixel_index, y_pixel_index, spec_index, frame_axis_index, pp_index
from .io.veronica import pixel_to_nm
from .core.scan import Scan, TimeScan
from .utils.static import nm_to_ir_wavenumbers

debug = 0

class WidgetBase():
    """A Base class that contains the most generic widgets for a class using
    allYouCanEat data."""
    def __init__(self, data=AllYouCanEat(), fig=None, ax=None,
                 central_wl=None, vis_wl=None, figsize=None):
        self.data = data
        self.data_base = np.zeros_like(self.data.data)
        self._fig = fig
        self._central_wl = central_wl
        self._vis_wl = vis_wl
        self._figsize = figsize
        self.children = [] # List of widgets to display
        self._init_widget()

    def __call__(self):
        """Use call to actually Render the widgets on the notebook."""
        from IPython.display import display
        self._configure_widgets()
        self._init_observer()
        self._update_figure()
        display(self.children)
        self.fig

    def _init_widget(self):
        """Init all widgets.

        """
        # A widget property starts with w_.
        # self.w_children is going to be displayed. All widgets that should be
        # visible must register with that.
        # Its not too bad if you have it widget more here then you need.
        # You then just don't.
        # Register it with self.children.
        import ipywidgets as wi

        self.w_folder = wi.Text(
            #layout = wi.Layout(max_width='100%')
        )
        self.w_file = wi.Select(
            descripion='Files',
            #layout = wi.Layout(max_width='100%')
        )
        self.w_base = wi.Select(description='Base')
        self.w_sub_base = wi.ToggleButton(description='Sub Baseline', width='110px')
        self.w_show_baseline = wi.Checkbox(description='Show Baseline',
                                        value=True)
        self.w_smooth_s = wi.IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=19, step=2,
        )
        self.w_Autoscale = wi.Checkbox(
            description="Autoscale",
            value=True,
        )
        self.w_y_pixel_range = wi.IntRangeSlider(
            continuous_update=False, description="y_pixels"
        )
        self.w_x_pixel_range = wi.IntRangeSlider(
            continuous_update=False, description="x_pixels",
            max = 1600, value=(400, 1200),
        )

        self.w_central_wl = wi.FloatText(
            description='central wl', value=self.central_wl, width='120px'
        )
        self.w_calib = wi.Dropdown(
            description='x-axis', options=['pixel', 'nm', 'wavenumber'],
            width='60px'
        )
        self.w_vis_wl = wi.FloatText(
            description='vis wl', value=self.vis_wl, width='120px'
        )
        self.w_pp_s = wi.SelectionSlider(
            continuous_update=False, description="pp_delay",
        )
        self.w_frame = wi.IntSlider(
            continuous_update=False, description="frame"
        )
        self.w_frame_median = wi.Checkbox(
            description='median',
        )
        self.w_pp_baseline_slider = wi.IntSlider(
            description='pp_delay index', continuous_update=False)
        self.w_frame_baseline = wi.IntSlider(
            description='frame', continuous_update=False)
        self.w_frame_base_med = wi.Checkbox(
            description='median', value=False)
        self.w_spec_base_slider = wi.IntSlider(
            description='spectrum', continuous_update=False)
        self.w_sum_over = wi.Dropdown(description='sum x-axis',
                                       options=('pp_delays', 'frames'),
                                      width='60px',)
        self.w_baseline_offset = wi.FloatText(description='Offset', value=0, widht = "60px")

        ### From here on the aligning boxers ###
        folder_box = wi.HBox([wi.Label("Folder", margin='0px 123px 0px 0px'), self.w_folder])
        self.w_data_box = wi.VBox([
            wi.Label("Data:"),
            folder_box,
            wi.HBox([
                wi.VBox([self.w_sub_base, self.w_show_baseline, self.w_Autoscale,]),
                self.w_file, self.w_base,
            ])
        ])
        #self.w_data_box.border = '1px black solid'
        self.w_data_box.margin = '8px 0px 8px 0px'

        self.w_signal_box = wi.VBox([
                wi.Label("Spectrum:"),
                wi.HBox([self.w_pp_s, self.w_smooth_s, self.w_frame_median]),
                wi.HBox([self.w_frame, self.w_y_pixel_range, self.w_x_pixel_range]),
        ])
        #self.w_signal_box.border = "1px black solid"
        self.w_signal_box.margin = "8px 0px 8px 0px"

        self.w_baseline_box = wi.VBox([
            wi.Label("Baseline:"),
            wi.HBox([
                self.w_pp_baseline_slider, self.w_frame_baseline,
                self.w_spec_base_slider, self.w_frame_base_med, ]),
                self.w_baseline_offset
            ])
        #self.w_baseline_box.border = "1px black solid"
        self.w_baseline_box.margin = "8px 0px 8px 0px"

        self.w_calib.margin = "0px 130px 0px 0px"

    def _configure_widgets(self):
        """Set all widget options. And default values.

        It must be save to recall this method at any time."""

        self.w_pp_s.options = self.data.pp_delays.tolist()
        if self.w_pp_s.value not in self.w_pp_s.options:
            self.w_pp_s.value = self.w_pp_s.options[0]
        if self.data.pp_delays.shape == (1,):
            self.w_pp_s.disabled = True
        else:
            self.w_pp_s.disabled = False

        self.w_frame.max = self.data.data.shape[frame_axis_index] - 1
        if self.data.data.shape[frame_axis_index] == 1:
            self.w_frame.disabled = True
        else:
            self.w_frame.disabled = False
        self.w_y_pixel_range.max = self.data.data.shape[y_pixel_index]
        if np.any(np.array(self.w_y_pixel_range.value) > self.w_y_pixel_range.max):
            self.w_y_pixel_range.value = self.w_y_pixel_range.min, self.w_y_pixel_range.max
        if self.data.data.shape[y_pixel_index] == 1:
            self.w_y_pixel_range.disabled = True
        else:
            self.w_y_pixel_range.disabled = False
        self.w_x_pixel_range.max = self.data.data.shape[x_pixel_index]
        if np.any(np.array(self.w_x_pixel_range.value) > self.w_x_pixel_range.max):
            self.w_x_pixel_range.value = self.w_x_pixel_range.min, self.w_x_pixel_range.max

        if isinstance(self.central_wl, type(None)):
            self.w_central_wl.value = 0
        else:
            self.w_central_wl.value = self.central_wl
        self.w_vis_wl.value = self.vis_wl
        self.w_pp_baseline_slider.max = self.data_base.shape[pp_index] - 1
        if self.w_pp_baseline_slider.max is 0:
            self.w_pp_baseline_slider.disabled = True
        else:
            self.w_pp_baseline_slider.disabled = False
        if self.w_pp_baseline_slider.value > self.w_pp_baseline_slider.max:
            self.w_pp_baseline_slider.value = 0
        self.w_frame_baseline.max = self.data_base.shape[frame_axis_index] - 1
        if self.w_frame_baseline.max is 0:
            self.w_frame_baseline.disabled = True
        else:
            self.w_frame_baseline.disabled = False
        if self.w_frame_baseline.value >  self.w_frame_baseline.max:
            self.w_frame_baseline.value = 0
        self.w_spec_base_slider.max = self.data_base.shape[y_pixel_index] - 1
        if self.w_spec_base_slider.max is 0:
            self.w_spec_base_slider.disabled = True
        else:
            self.w_spec_base_slider.disabled = False
        if self.w_spec_base_slider.value > self.w_spec_base_slider.max:
            self.w_spec_base_slider.value = 0

        self._toggle_central_wl()
        self._toggle_vis_wl()
        self._toggle_frame_slider()
        self._toggle_sum_over()

    def _toggle_vis_wl(self, new=None):
        """Toggle the vis wl text box according to calibration axis.

        The new keyword exists, so if can also server as a callback function."""
        if self.w_calib.value == 'wavenumber':
            self.w_vis_wl.disabled = False
        else:
            self.w_vis_wl.disabled = True

    def _toggle_central_wl(self, new=None):
        """Toggle the central wl text box according to calibration axis.

        The new keyword exists, so if can also server as a callback function."""
        if self.w_calib.value == 'pixel' or self.data._type == 'spe':
            self.w_central_wl.disabled = True
        else:
            self.w_central_wl.disabled = False

    def _toggle_frame_slider(self, new=None):
        """Toggle frame slider according to number of frames.

        The new keyword exists, so if can also server as a callback function."""
        if self.w_frame_median.value:
            self.w_frame.disabled = True
        else:
            self.w_frame.disabled = False

    def _toggle_frame_base_slider(self, new=None):
        """Toggle frame slider of the base/background according to frames.

        The new keyword exists, so if can also server as a callback function."""
        if self.w_frame_base_med.value:
            self.w_frame_baseline.disabled = True
        else:
            self.w_frame_baseline.disabled = False

    def _toggle_sum_over(self, new=None):
        if self.data.frames is 1:
            self.w_sum_over.value = "pp_delays"
            self.w_sum_over.disabled = True
            return
        if self.data.pp_delays.shape[0] is 1:
            self.w_sum_over.value = "frames"
            self.w_sum_over.disabled = True
            return
        self.w_sum_over.disabled = False

    def _init_figure(self):
        """Init figure.

        This will be most likely  be overwritten by the children."""
        if not self.fig:
            self._fig = plt.gcf()

        if len(self.axes) is 0:
            self._fig.add_subplot(111)

    def _update_figure(self):
        """Update figure with changes"""
        # MUST OVERWRITE THIS
        pass

    def _update_figure_callback(self, new):
        """A callback version of _update_figer for usage in observers."""
        self._update_figure()

    def _update_data(self, new):
        """Gets called when new data is selected in the gui."""
        fname = self.w_folder.value + "/" + self.w_file.value
        self.data = AllYouCanEat(fname)
        self._central_wl = None
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        self._unobserve()
        self._configure_widgets()
        self._init_observer()
        self._update_figure()

    def _on_base_changed(self, new):
        """Called when baseline is changed."""
        fname = self.w_folder.value + "/" + self.w_base.value
        # If we have already loaded the data to ram for the baseline,
        # we just copy it from there. Thats a lot faster then reading
        # it from the HDD again.
        if self.w_base.value is self.w_file.value:
            if debug:
                print("Copied new baseline data from ram.")
            self.data_base = self.data.data.copy()
        else:
            self.data_base = AllYouCanEat(fname).data
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        self._unobserve()
        self._configure_widgets()
        self._init_observer()
        self._update_figure()

    def _init_observer(self):
        """Set all observer of all subwidgets."""
        # This registers the callback functions to the gui elements.
        # After a call of _init_observer, the gui elements start to
        # actually do something.
        self.w_folder.on_submit(self._on_folder_submit)
        self.w_file.observe(self._update_data, 'value')
        self.w_file.observe(self._update_figure_callback, 'value')
        self.w_calib.observe(self._toggle_vis_wl, 'value')
        self.w_calib.observe(self._toggle_central_wl, 'value')
        self.w_calib.observe(self._update_figure_callback, 'value')
        self.w_pp_s.observe(self._update_figure_callback, 'value')
        self.w_frame.observe(self._update_figure_callback, 'value')
        self.w_frame_median.observe(self._toggle_frame_slider, 'value')
        self.w_frame_median.observe(self._update_figure_callback, 'value')
        self.w_y_pixel_range.observe(self._update_figure_callback, 'value')
        self.w_x_pixel_range.observe(self._update_figure_callback, 'value')
        self.w_vis_wl.observe(self._update_figure_callback, 'value')
        self.w_central_wl.observe(self._update_figure_callback, 'value')
        self.w_Autoscale.observe(self._update_figure_callback, 'value')
        self.w_smooth_s.observe(self._update_figure_callback, 'value')
        self.w_base.observe(self._on_base_changed, 'value')
        self.w_sub_base.observe(self._update_figure_callback, 'value')
        self.w_show_baseline.observe(self._update_figure_callback, 'value')
        self.w_pp_baseline_slider.observe(self._update_figure_callback, 'value')
        self.w_frame_baseline.observe(self._update_figure_callback, 'value')
        self.w_spec_base_slider.observe(self._update_figure_callback, 'value')
        self.w_frame_base_med.observe(self._update_figure_callback, 'value')
        self.w_frame_base_med.observe(self._toggle_frame_base_slider, 'value')
        self.w_sum_over.observe(self._update_figure_callback, "value")
        self.w_baseline_offset.observe(self._update_figure_callback, "value")

    # TODO Refactor this with a lisf of figure updating
    # widgets and then unobserver only the figure updates.
    def _unobserve(self):
        """Turn off all the observers."""
        self.w_folder.unobserve_all()
        self.w_file.unobserve_all()
        self.w_calib.unobserve_all()
        self.w_pp_s.unobserve_all()
        self.w_frame.unobserve_all()
        self.w_frame_median.unobserve_all()
        # Hack untill i split this up in dedicated method
        # to unobserver only the figure callbacks.
        self.w_y_pixel_range.unobserve(self._update_figure_callback, "value")
        self.w_x_pixel_range.unobserve_all()
        self.w_vis_wl.unobserve_all()
        self.w_central_wl.unobserve_all()
        self.w_Autoscale.unobserve_all()
        self.w_smooth_s.unobserve_all()
        self.w_base.unobserve_all()
        self.w_sub_base.unobserve_all()
        self.w_show_baseline.unobserve_all()
        self.w_pp_baseline_slider.unobserve_all()
        self.w_frame_baseline.unobserve_all()
        self.w_spec_base_slider.unobserve_all()
        self.w_frame_base_med.unobserve_all()
        self.w_sum_over.unobserve_all()
        self.w_baseline_box.unobserve_all()

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._fig.axes

    #TODO Find a better name.
    def _prepare_x_data(self, data, wavelength=None, wavenumber=None):
        """Apply transformations to data, that don't change the shape"""

        # Need pixel as a possible default value.
        pixel = np.arange(data.shape[x_pixel_index])
        central_wl = self.w_central_wl.value
        vis_wl = self.w_vis_wl.value

        if self.w_calib.value == 'pixel':
            x = pixel
        if self.w_calib.value == 'nm':
            x = wavelength
            if isinstance(wavelength, type(None)):
                x = np.arange(len(pixel))
            if self.w_central_wl.value != 0:
                x = pixel_to_nm(pixel, central_wl)
        if self.w_calib.value == 'wavenumber':
            x = wavenumber
            if isinstance(wavenumber, type(None)):
                x = pixel[::-1]
            if central_wl == 0 and vis_wl != 0 and not isinstance(wavelength, type(None)):
                x = nm_to_ir_wavenumbers(wavelength, vis_wl)
            if central_wl != 0 and vis_wl != 0:
                nm = pixel_to_nm(pixel, central_wl)
                x = nm_to_ir_wavenumbers(nm, vis_wl)
        return x

    @property
    def x(self):
        """X data of the plot *Signal* plot.

        Shape changing transformations belong in here."""

        # Its not guaranteed, that wavelength and wavenumber exist.
        wavelength = getattr(self.data, 'wavelength', None)
        wavenumber = getattr(self.data, 'wavenumber', None)
        #x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
        #                                 self.data.data.shape[x_pixel_index])
        ret = self._prepare_x_data(
            self.data.data,
            wavelength,
            wavenumber
        )
        return ret

    def _prepare_y_data(self, data):
        """Non shape changing transformations."""
        #TODO Test if copy is still needed.
        y = data.copy()
        if self.w_sub_base.value:
            y -= np.ones_like(y) * self.y_base
        return y

    @property
    def y(self):
        """Y data of the *Signal* plot."""

        #TODO the ppdelay slider needs to be a custom slider.
        pp_delays = getattr(self.data, 'pp_delays')
        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         self.data.data.shape[y_pixel_index])
        pp_delay_index = np.where(
            self.w_pp_s.value == pp_delays)[0][0]

        # TODO add the possibility to select frame regions.
        ret =  self._prepare_y_data(self.data.data)[pp_delay_index, :, :, :]
        if self.w_frame_median.value:
            ret = np.median(ret, frame_axis_index)
        else:
            ret = ret[
                self.w_frame.value,
                :,
                :
            ]
        ret = ret[y_slice, :]
        # Must be done here, because it works only on 2d data.
        if self.w_smooth_s.value != 1:
            ret = medfilt2d(ret, (1, self.w_smooth_s.value))
        return ret.T

    @property
    def x_base(self):
        """x data of the baseline in the *Signal* plot"""
        wavelength = getattr(self.data, 'wavelength', None)
        wavenumber = getattr(self.data, 'wavenumber', None)
        x = self._prepare_x_data(
            self.data_base,
            wavelength,
            wavenumber
        )
        return x

    @property
    def y_base(self):
        """y data of the baseline in the *Signal* plot."""

        # TODO this is a hack, because I cant select frame regions yet
        frame_slice = self.w_frame_baseline.value
        if self.w_frame_base_med.value:
            data = self.data_base[
                self.w_pp_baseline_slider.value,
                :,
                self.w_spec_base_slider.value,
                :]
            y = np.median(data, 0) + self.w_baseline_offset.value
        else:
            y = self.data_base[
                    self.w_pp_baseline_slider.value,
                    frame_slice,
                    self.w_spec_base_slider.value,
                    :] + self.w_baseline_offset.value
        return y.T

    @property
    def central_wl(self):
        """The central wavelength used for x data calibration."""
        return self._central_wl

    @property
    def vis_wl(self):
        """The wavelength of the visible.

        The visible wavelength is used as upconversion number during them
        calculation of the wavenumber values of the x axis of the *Signal* plot."""
        if self._vis_wl == None:
            return 0
        return self._vis_wl

    def _on_folder_submit(self, new):
        """Called when folder is changed."""
        fnames = _filter_fnames(self.w_folder.value)

        if debug:
            print("_on_folder_submit_called")
        if debug > 1:
            print("fnames:", fnames)

        self.w_file.options = fnames
        self.w_base.options = self.w_file.options


class SpecAndBase(WidgetBase):
    """A widget that allows for selection of a spectrum and baseline plus visualization."""

    def __init__(self, figsize=(8 , 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

    def __call__(self):
        super().__call__()

    def _init_widget(self):
        """Init the widgets that are to be shown."""
        import ipywidgets as wi
        super()._init_widget()
        # This allows the data to be used for normalization from start on
        self.data.data += 1
        self.w_x_pixel_range.layout.visibility = 'hidden'
        self.children = wi.VBox([
                self.w_data_box,
                self.w_signal_box,
                self.w_baseline_box,
                wi.HBox([self.w_calib, self.w_central_wl, self.w_vis_wl]),
        ])

    def _init_figure(self):
        """Init the fiures and axes"""
        if not self._fig:
            self._fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        # This allows for redrawing the axis on an already existing figure.
        elif self._fig and len(self.axes) is not 1:
            self._fig.set_size_inches(self._figsize, forward=True)
            self._ax = self._fig.add_subplot(111)

    def _update_figure(self):
        """Is called on all gui element changes.

        This function renders the plot. When ever you want to make changes
        visible in the figure you need to run this."""
        self._init_figure()
        ax = self.axes[0]
        ax.clear()
        ax.plot(self.x, self.y, label='Spectrum')
        if self.w_show_baseline.value:
            ax.plot(self.x_base, self.y_base, label='Baseline')
        ax.legend(framealpha=0.5)
        if self.w_Autoscale.value:
            self._ax_xlim = ax.get_xlim()
            self._ax_ylim = ax.get_ylim()
        if not isinstance(self._ax_xlim, type(None)):
            ax.set_xlim(*self._ax_xlim)
        if not isinstance(self._ax_ylim, type(None)):
            ax.set_ylim(*self._ax_ylim)


class SpecAndSummed(WidgetBase):
    def __init__(self, central_wl=None, vis_wl=810, figsize=(10, 4), **kwargs):
        """Plotting gui based on the AllYouCanEat class as a data backend.

        Parameters
        ----------
        data : Optional, AllYouCanEat obj.
            Default dataset to start with. If None is given, an empty one is created.
        fig: Optional, matplotlib figure
            The figure to draw on.
            Defaults to create a new one.
        ax: Optional, matplotlib axes. The axes to draw on.
            Defaults to create a new one.
        central_wl: float
            Central wavelength of the camera to start with.
            If none is given, it tryes to find out from by investigating the
            metadata.
        vis_wl: float
            Wavelength of the visible to begin with.

        Example:
        -------
        test = AllYouCanPlot()
        test()
        # Type the Folder you want to investigate in the folder Text box and press RETURN.
        # A list of selectable files will appear on the right side.
        """
        super().__init__(central_wl=central_wl, vis_wl=vis_wl, figsize=figsize,
                         **kwargs)
        self._ax_xlim = None
        self._ax_ylim = None

    def _init_figure(self):
        """Init the two axis figure."""
        if not self._fig:
            self._fig, ax = plt.subplots(1, 2, figsize=self._figsize)
        # This allows for redrawing the axis on an already existing figure.
        elif self._fig and len(self.axes) is not 2:
            self._fig.set_size_inches(self._figsize, forward=True)
            [self._fig.add_subplot(121), self._fig.add_subplot(122)]

    def _init_widget(self):
        """Init all widgets that are to be drawn."""
        import ipywidgets as wi
        super()._init_widget()
        # self.children is the widget we are rendering up on call.

        self.children = wi.VBox([
                self.w_data_box,
                self.w_signal_box,
                self.w_baseline_box,
                wi.HBox([self.w_calib, self.w_central_wl, self.w_vis_wl]),
                self.w_sum_over,
        ])

    def _update_figure(self):
        """Update the figure of the gui.

        Gets called on all button changes."""
        self._init_figure()
        fontsize = 8
        ax = self.axes[0]
        ax.clear()
        ax.plot(self.x, self.y, label='Spectrum')
        if self.w_show_baseline.value:
            ax.plot(self.x_base, self.y_base, label='Baseline')
        ax.vlines(self.x_vlines, *ax.get_ylim(),
                  linestyle="dashed")
        ax.legend(framealpha=0.5)
        ax.set_xlabel(self.w_calib.value)
        ax.set_title('Spectrum')
        # Buffer ax_{x,y}_lim so we can reuse it later
        if self.w_Autoscale.value:
            self._ax_xlim = ax.get_xlim()
            self._ax_ylim = ax.get_ylim()
        if not isinstance(self._ax_xlim, type(None)):
            ax.set_xlim(*self._ax_xlim)
        if not isinstance(self._ax_ylim, type(None)):
            ax.set_ylim(*self._ax_ylim)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.set_title("Signal")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

        ax = self.axes[1]
        ax.clear()
        sum_y = self.sum_y
        lines0_1 = ax.plot(self.sum_x, sum_y)
        points0_1 = ax.plot(self.sum_x, sum_y, 'o')
        for i in range(len(lines0_1)):
            points = points0_1[i]
            color = lines0_1[i].get_color()
            points.set_color(color)
        if "pp_delays" in self.w_sum_over.value:
            ax.set_xlabel('pp delay / fs')
        else:
            ax.set_xlabel("# frame")
        ax.set_ylabel('Sum')
        ax.set_title('Summed')
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()

        self._fig.canvas.draw()
        for ax in self.axes:
            ax.figure.canvas.draw()

    @property
    def central_wl(self):
        """Central wl used for x axis calibration of the *Spectrum* axis."""
        if self._central_wl == None:
            if self.data.metadata.get('central_wl') != 0:
                self._central_wl = self.data.metadata.get('central_wl')
        return self._central_wl

    @property
    def sum_y(self):
        """y data of the summed plot."""
        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         self.data.data.shape[y_pixel_index])
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         self.data.data.shape[x_pixel_index])

        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.w_pp_s.value == pp_delays)[0][0]

        y = self._prepare_y_data(self.data.data)
        y = y[:, :, y_slice, x_slice]


        if 'pp_delays' in self.w_sum_over.value:
            if self.w_frame_median.value:
                y = np.median(y, frame_axis_index)
            else:
                y = y[:, self.w_frame.value, :, :]
        if 'frames' in self.w_sum_over.value:
            y = y[pp_delay_index, :, :, :]

        if self.w_smooth_s.value is not 1:
            y = medfilt(y, (1, 1, self.w_smooth_s.value))

        return y.sum(x_pixel_index)

    @property
    def sum_x(self):
        """x data of the summed plot."""
        #print(self.w_sum_over.value)
        if 'pp_delays' in self.w_sum_over.value:
            return self.data.pp_delays
        elif 'frames' in self.w_sum_over.value:
            return np.arange(self.data.frames)
        raise NotImplementedError('got %s for w_sum_over' % self.w_sum_over.value)

    @property
    def x_vlines(self):
        ret = [self.x[self.w_x_pixel_range.value[0]],
               self.x[self.w_x_pixel_range.value[1] - 1]]
        return ret

    def _on_xlim_changed(self, new=None):
        """Callback for the *Signal* axis."""
        # Called when the xlim of the *Signal* plot is changed
        self._ax_xlim = self.axes[0].get_xlim()

    def _on_ylim_changed(self, new=None):
        # Called when the ylim of the *Signal* plot is cahnged
        self._ax_ylim = self.axes[0].get_ylim()

# TODO Find Better name
class Normalized(SpecAndSummed):
    """A simplified version of SpecAndSummed.

    This Widget is used to visualize data after normalization."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.w_show_baseline.value = False
        self.children = wi.VBox([
            #self.w_reload,
            wi.HBox([self.w_pp_s, self.w_smooth_s, self.w_Autoscale]),
            wi.HBox([self.w_frame, self.w_y_pixel_range, self.w_x_pixel_range, self.w_frame_median,
                  #self.w_x_pixel_range
            ]),
            wi.HBox([self.w_calib, self.w_sum_over]),
            self.w_central_wl, self.w_vis_wl,
        ])

class ImgView(WidgetBase):
    """A Class to view full spe images."""
    def __init__(self, *args, figsize=(8,6), **kwargs):
        super().__init__(*args, figsize=figsize, **kwargs)

    def _init_figure(self):
        if not self._fig:
            self._fig = plt.figure(self._figsize)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3,1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)
        elif self._fig and len(self.axes) is not 3:
            self._fig.set_size_inches(self._figsize, forward=True)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3,1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)


    def _update_figure(self):
        self._init_figure()
        view_data = self.data.data[self.w_pp_s.value, self.w_frame.value]
        ax = self.axes[0]
        plt.sca(ax)
        ax.clear()
        img = ax.imshow(view_data, interpolation=self.w_interpolate.value,
                  origin="lower", aspect="auto")
        plt.colorbar(img)

        axl = self.axes[1]
        axl.clear()
        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         self.data.data.shape[y_pixel_index])
        view_data2 = self.data.data[self.w_pp_s.value, self.w_frame.value, y_slice].sum(y_pixel_index)
        axl.plot(view_data2)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.w_smooth_s.visible = False
        self.w_smooth_s.disabled  = True
        self.w_pp_s.visible = False
        self.w_pp_s.disabled = True
        self.w_interpolate = wi.Dropdown(
            description="Interpolation",
            options=('none', 'nearest', 'bilinear', 'bicubic',
                     'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
                     'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                     'lanczos'),
            value = "nearest",
        )
        self.children = wi.VBox([
            self.w_signal_box,
            wi.HBox([self.w_calib, self.w_central_wl, self.w_vis_wl]),
            self.w_interpolate,
        ])

    def _init_observer(self):
        super()._init_observer()
        self.w_interpolate.observe(self._update_figure_callback, "value")


##### Helper function
def _filter_fnames(folder_path):
    """Return list of known files in a folder."""

    fnames = np.sort(glob(os.path.normcase(folder_path + '/*' )))
    # Only .dat and .spe in known
    mask = [a or b for a, b in zip([".dat" in s for s in fnames], [".spe" in s for s in fnames])]
    fnames = fnames[np.where(mask)]
    # Remove AVG
    fnames = fnames[np.where(["AVG" not in s for s in fnames])]
    fnames = [os.path.split(elm)[1] for elm in fnames]
    return fnames


def _slider_range_to_slice(range_value_tuple, max):
    """Transform a tuple into a slice accounting for overlapping"""
    # This can happen if one links
    # sliders but can not guarantee that they
    # refer to the same data.
    if any(range_value_tuple) > max:
            if range_value_tuple[0] > max:
                range_value_tuple[0] = max
            range_value_tuple[1] = range_value_tuple[0]

    if range_value_tuple[0] == range_value_tuple[1]:
        if range_value_tuple[0] < max:
            range_value_tuple = range_value_tuple[0], range_value_tuple[0] + 1
        else:
            range_value_tuple = range_value_tuple[1] - 1, range_value_tuple[1]

    return slice(*range_value_tuple)

#### End of helper functions
