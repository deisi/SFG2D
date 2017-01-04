"""Module for widget classes."""

import warnings
import os
from glob import glob
import numpy as np
from scipy.signal import medfilt2d, medfilt
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
        from IPython.display import display
        self._configure_widgets()
        self._init_observer()
        self._update_figure()
        display(self.children)
        self.fig

    def _init_widget(self):
        """Init all widgets."""
        # A widget property starts with w_.
        # self.w_children is going to be displayed. All widgets that should be
        # visible must register with that.
        import ipywidgets as wi

        self.w_folder = wi.Text(description="Folder")
        self.w_file = wi.Select(
            descripion='Files',
            layout = wi.Layout(width='70%', max_width='100%')
        )
        self.w_base = wi.Select(description='Base')
        self.w_sub_base = wi.ToggleButton(description='Sub Baseline')
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
            continuous_update=False, description="x_pixels"
        )

        self.w_central_wl = wi.FloatText(
            description='Central Wl', value=self.central_wl, width='120px'
        )
        self.w_calib = wi.Dropdown(
            description='Calibration', options=['pixel', 'nm', 'wavenumber'],
            width='60px'
        )
        self.w_vis_wl = wi.FloatText(
            description='Vis Wl', value=self.vis_wl, width='120px'
        )
        self.w_pp_s = wi.SelectionSlider(
            continuous_update=False, description="pp_delay",
        )
        self.w_frame = wi.SelectionSlider(
            continuous_update=False, description="frame"
        )
        self.w_frame_median = wi.Checkbox(
            description='median',
        )
        self.w_pp_baseline_slider = wi.SelectionSlider(
            description='Baseline pp_delay index', continuous_update=False)
        self.w_frame_baseline = wi.SelectionSlider(
            description='Baseline Frame', continuous_update=False)
        self.w_frame_base_med = wi.Checkbox(
            description='median', value=False)
        self.w_spec_base_slider = wi.SelectionSlider(
            description='Baseline Spectrum', continuous_update=False)
        self.w_filebox = wi.HBox([self.w_file, self.w_base,
                               wi.VBox([self.w_sub_base, self.w_show_baseline, self.w_Autoscale,])
        ])

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

        self.w_frame.options = list(range(0, self.data.data.shape[frame_axis_index]))
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
        self.w_x_pixel_range.value = self.w_x_pixel_range.min, self.w_x_pixel_range.max

        if isinstance(self.central_wl, type(None)):
            self.w_central_wl.value = 0
        else:
            self.w_central_wl.value = self.central_wl
        self.w_vis_wl.value = self.vis_wl
        self.w_pp_baseline_slider.options = list(range(self.data_base.shape[pp_index]))
        if self.w_pp_baseline_slider.value not in self.w_pp_baseline_slider.options:
            self.w_pp_baseline_slider.value = 0
        self.w_frame_baseline.options = list(range(self.data_base.shape[frame_axis_index]))
        if self.w_frame_baseline.value not in self.w_frame_baseline.options:
            self.w_frame_baseline.value = 0
        self.w_spec_base_slider.options = list(range(self.data_base.shape[y_pixel_index]))
        if self.w_spec_base_slider.value not in self.w_spec_base_slider.options:
            self.w_spec_base_slider.value = 0

        self._toggle_central_wl()
        self._toggle_vis_wl()
        self._toggle_frame_slider()

    def _toggle_vis_wl(self, new=None):
        if self.w_calib.value == 'wavenumber':
            self.w_vis_wl.disabled = False
        else:
            self.w_vis_wl.disabled = True

    def _toggle_central_wl(self, new=None):
        if self.w_calib.value == 'pixel' or self.data._type == 'spe':
            self.w_central_wl.disabled = True
        else:
            self.w_central_wl.disabled = False

    def _toggle_frame_slider(self, new=None):
        '''toggle frame slider on or of '''
        if self.w_frame_median.value:
            self.w_frame.disabled = True
        else:
            self.w_frame.disabled = False

    def _init_figure(self):
        """Init initial figure."""
        if not self.fig:
            self._fig = plt.gcf()

        if len(self.axes) is 0:
            self._fig.add_subplot(111)

    def _update_figure(self):
        """Update figure with changes"""
        # MUST OVERWRITE THIS
        pass

    def _update_figure_callback(self, new):
        self._update_figure()

    def _update_data(self, new):
        fname = self.w_folder.value + "/" + self.w_file.value
        self.data = AllYouCanEat(fname)
        self._central_wl = None
        self._configure_widgets()

    def _init_observer(self):
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

    def _unobserve(self):
        pass

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._fig.axes

    def _prepare_x_data(self, data, wavelength=None, wavenumber=None):
        """Apply gui transformations to given data."""
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         data.shape[x_pixel_index])
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
        return x[x_slice]

    @property
    def x(self):
        """X data of the plot"""
        wavelength = getattr(self.data, 'wavelength', None)
        wavenumber = getattr(self.data, 'wavenumber', None)
        return self._prepare_x_data(
            self.data.data,
            wavelength,
            wavenumber
        )

    def _prepare_y_data(self, data, pp_delays):

        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         data.shape[y_pixel_index])
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         data.shape[x_pixel_index])
        pp_delay_index = np.where(
            self.w_pp_s.value == pp_delays)[0][0]

        y = data[:, :, y_slice, x_slice].copy()
        if self.w_sub_base.value:
            y -= np.ones_like(y) * self.y_base
        # TODO add the possibility to select frame regions.
        if self.w_frame_median.value:
            y = y[pp_delay_index, :, :, :]
            y = np.median(y, 0)
        else:
            y = y[
                pp_delay_index,
                self.w_frame.value,
                :,
                :
            ]
        if self.w_smooth_s.value != 1:
            y = medfilt2d(y, (1, self.w_smooth_s.value))
        return y.T

    @property
    def y(self):
        """Y data used in the plot."""
        pp_delays = getattr(self.data, 'pp_delays')
        return self._prepare_y_data(self.data.data, pp_delays)

    @property
    def x_base(self):
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
        # y_slice needed to have a consistent interface between background and spectra data
        # y_slice = _slider_range_to_slice(
        #     self.w_y_pixel_range.value,
        #     self.data_base.shape[y_pixel_index]
        # )
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         self.data_base.shape[x_pixel_index])

        # TODO this is a hack, because I cant select frame regions yet
        frame_slice = self.w_frame_baseline.value
        if self.w_frame_base_med.value:
            data = self.data_base[
                self.w_pp_baseline_slider.value,
                :,
                self.w_spec_base_slider.value,
                x_slice]
            y = np.median(data, 0)
        else:
            y = self.data_base[
                    self.w_pp_baseline_slider.value,
                    frame_slice,
                    self.w_spec_base_slider.value,
                    x_slice]
        return y.T

    @property
    def central_wl(self):
        return self._central_wl

    @property
    def vis_wl(self):
        if self._vis_wl == None:
            return 0
        return self._vis_wl

    def _on_folder_submit(self, new):
        fnames = _filter_fnames(self.w_folder.value)

        if debug:
            print("_on_folder_submit_called")
        if debug > 1:
            print("fnames:", fnames)

        self.w_file.options = fnames
        self.w_base.options = self.w_file.options

    def _on_base_changed(self, new):
        fname = self.w_folder.value + "/" + self.w_base.value
        self.data_base = AllYouCanEat(fname).data
        self._configure_widgets()
        self._update_figure()

class SpecAndBase(WidgetBase):

    def __init__(self, figsize=(8 , 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.children = wi.VBox([
                self.w_folder,
                self.w_filebox,
                wi.HBox([self.w_pp_s, self.w_smooth_s]),
                wi.HBox([self.w_frame, self.w_y_pixel_range, self.w_frame_median,
                      #self.w_x_pixel_range
                ]),
                wi.HBox([self.w_pp_baseline_slider, self.w_frame_baseline,
                          self.w_frame_base_med, ]),
                self.w_spec_base_slider,
                self.w_calib, self.w_central_wl, self.w_vis_wl,
        ])

    def _init_figure(self):
        """Init the fiures and axes"""
        if not self._fig:
            self._fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        elif self._fig and len(self.axes) is not 1:
            self._fig.set_size_inches(self._figsize, forward=True)
            self._ax = self._fig.add_subplot(111)

    def _update_figure(self):
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
        if not self._fig:
            self._fig, ax = plt.subplots(1, 2, figsize=self._figsize)
        elif self._fig and len(self.axes) is not 2:
            self._fig.set_size_inches(self._figsize, forward=True)
            [self._fig.add_subplot(121), self._fig.add_subplot(122)]

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.children = wi.VBox([
                self.w_folder,
                self.w_filebox,
                wi.HBox([self.w_pp_s, self.w_smooth_s, self.w_frame_median]),
                wi.HBox([self.w_frame, self.w_y_pixel_range, self.w_x_pixel_range]),
                wi.HBox([self.w_pp_baseline_slider, self.w_frame_baseline,
                      self.w_frame_base_med,]),
                self.w_spec_base_slider,
                self.w_calib, self.w_central_wl, self.w_vis_wl,
        ])

    def _update_figure(self):
        """Update figure on page 0."""
        self._init_figure()
        fontsize = 8
        ax = self.axes[0]
        ax.clear()
        ax.plot(self.x, self.y, label='Spectrum')
        if self.w_show_baseline.value:
            ax.plot(self.x_base, self.y_base, label='Baseline')
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
        ax.set_xlabel('pp delay / fs')
        ax.set_ylabel('Sum')
        ax.set_title('Summed')
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()

        self._fig.canvas.draw()
        for ax in self.axes:
            ax.figure.canvas.draw()

    @property
    def central_wl(self):
        if self._central_wl == None:
            if self.data.metadata.get('central_wl') != 0:
                self._central_wl = self.data.metadata.get('central_wl')
        return self._central_wl

    @property
    def norm_y(self):
        """y-data of the normalization plot."""
        pass

    @property
    def norm_x(self):
        """X data of the normalization data."""
        pass

    @property
    def sum_y(self):
        """y data of the summed plot."""
        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         self.data.data.shape[y_pixel_index])
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         self.data.data.shape[x_pixel_index])

        if self.w_frame_median.value:
            # TODO Maybe merge this with y property. They are redundant to some
            # extend.
            data = np.median(self.data.data, frame_axis_index)
            y = data[:, y_slice, x_slice].sum(x_pixel_index)
        else:
            y = self.data.data[:, self.w_frame.value, y_slice, x_slice].sum(x_pixel_index)
        return y

    @property
    def sum_x(self):
        """x data of the summed plot."""
        return self.data.pp_delays

    def _on_xlim_changed(self, new=None):
        self._ax_xlim = self.axes[0].get_xlim()

    def _on_ylim_changed(self, new=None):
        self._ax_ylim = self.axes[0].get_ylim()




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


#### End of helper function




########################################
# I plan to remove all below this points
########################################
class MyBqPlot():
    """Class for Bqolot setup. """

    def fig_init(self, title='',x_label='', y_label=''):
        """Init an empty bqplot figure"""
        import bqplot as bq
        from ipywidgets import ToggleButton, BoundedIntText

        x_sc = bq.LinearScale()
        y_sc = bq.LinearScale()

        line = bq.Lines(scales={'x':x_sc, 'y':y_sc})

        ax_x = bq.Axis(scale=x_sc,
                    label=x_label)
        ax_y = bq.Axis(scale=y_sc,
                    label=y_label, orientation='vertical')

        # Zoom only y-scale
        def py_update(new):
            if tb_py.value:
                py = bq.PanZoom(scales={'y': [y_sc]})
                fig.interaction = py
            else:
               fig.interaction.close()
        tb_py = ToggleButton(description="Zoom-Y")
        tb_py.observe(py_update, "value")

        # Smooth data here because it only effects the way the 
        # data is shown, it has no deeper physical meaning
        # and doesnt interfere with the data containing classes
        smooth_width = BoundedIntText(
            value=0, min=0, max=100,
            description = 'Smooth',
            tooltip = 'Size of the windows used for rolling median to smooth data'
            )
        smooth_width.observe(self.fig_update, "value")

        fig = bq.Figure(marks=[line], axes=[ax_x, ax_y], 
                     title=title)

        tb = bq.Toolbar(figure=fig)

        fig.pyplot = tb
        fig.tb_py = tb_py
        fig.smooth_width = smooth_width

        return fig

    def fig_update(self, *args, **kwargs):
        """Update bqplot figure with new data"""
        # use index of data to set labels
        if isinstance(self.data, type(None)):
            warnings.warn("No data to update figure with in %s" % self)
            return

        x_label = self.data.index.name
        if x_label == 'wavenumber':
            x_label += ' in 1/cm'

        self.fig.axes[0].label = x_label

        self.fig.marks[0].x = self.data.index
        if self.fig.smooth_width.value == 0:
            self.fig.marks[0].y = self.data.transpose()
        else:
            self.fig.marks[0].y = self.data.rolling(self.fig.smooth_width.value).median().transpose()

class DataImporter(MyBqPlot):
    data = None # Data presented by the widget
    scan = None # Full data used by the widget
    base = None # Full Baseline data
    # List of widget names used to obtain widget status
    _widgets = (
        'fpath_selector', 'fbase_selector',
        'spec_selector', 'sub_baseline_toggle'
    )

    def __init__(self, ffolder, fpath_selector,
                 fbase_selector,  pp_delay_slider,
                 spec_selector, sub_baseline_toggle):
        '''
        fpath_seeelector:
            Selection widget for fpath

        fbase_selector:
            Selection widget for the baseline

        pp_delay_slider:
            Slider for Pump-Probe Timedelay

        spec_selector:
            Selection widget for the spectrum of interest

        sub_baseline_toggle:
            Toggle button for substracting baseline
        '''
        # Init Properties
        self.ffolder = ffolder
        self.pp_delay_slider = pp_delay_slider
        self.fpath_selector = fpath_selector
        self.fbase_selector = fbase_selector
        self.spec_selector = spec_selector
        self.sub_baseline_toggle = sub_baseline_toggle

    def __call__(self, title='', x_label='', y_label=''):
        """ create the widget by calling it."""
        # Hackaround for not automatic valueupdate problem
        # when ffolder is changed on the fly
        from IPython.display import display
        from ipywidgets import HBox, VBox

        try:
            fpath = os.path.join(self.ffolder, self.fpath_selector.value)
            fbase = os.path.join(self.ffolder, self.fbase_selector.value)
        except TypeError:
            fpath = self.ffolder
            fbase = self.ffolder
        if not os.path.isfile(fpath) or not os.path.isfile(fbase):
            warnings.warn('File does not exist. Skipping update\n%s\n%s'
                          % (fpath, fbase))
            self.fig = self.fig_init(title, x_label, y_label)
        # Init the Widget
        else:
            self.get(fpath, fbase)
            self.update_data()
            #self.update_scan()
            self.fig = self.fig_init(title, x_label, y_label)
            self.fig_update()

        self.observe()

        # Display the Widget
        display(
            HBox([self.fpath_selector, self.fbase_selector]),
            HBox([self.pp_delay_slider, self.spec_selector]),
            HBox([self.sub_baseline_toggle, self.fig.smooth_width]),
            VBox([self.fig, self.fig.pyplot, self.fig.tb_py])
        )

    def observe(self):
        """ Init the observers """
        self.fpath_selector.observe(self.update, 'value')
        self.fpath_selector.observe(self.update_data, 'value')
        self.fpath_selector.observe(self.fig_update, 'value')

        self.fbase_selector.observe(self.update, 'value')
        self.fbase_selector.observe(self.update_data, 'value')
        self.fbase_selector.observe(self.fig_update, 'value')

        self.pp_delay_slider.observe(self.update_data, 'value')
        self.pp_delay_slider.observe(self.fig_update, 'value')

        self.spec_selector.observe(self.update_data, 'value')
        self.spec_selector.observe(self.fig_update, 'value')

        self.sub_baseline_toggle.observe(self.update_scan, 'value')
        self.sub_baseline_toggle.observe(self.fig_update, 'value')

    def unobserve(self):
        """ Unobservers all widgets"""
        for widget_name in self._widgets:
            widget = getattr(self, widget_name)
            widget.unobserve_all()

    def get(self, fpath, fbase):
        """Get Data from HDD

        fpath: path to data

        fbase: path to background data"""
        data = read_auto(fpath)
        self.scan = data
        if isinstance(data, TimeScan):
            self.pp_delay_slider.options = list(data.pp_delays)
            _exp_data = data.med.ix[list(data.pp_delays)[0]]
        else:
            self.pp_delay_slider.options = [0]
            self.pp_delay_slider.value = 0
            _exp_data = data.med

        base = read_auto(fbase)
        if isinstance(base, TimeScan) and not isinstance(data,  TimeScan):
            if 0 in base.pp_delays:
                base = base.med.ix[0]
            else:
                base = base.med.ix[base.pp_delays[0]]
            self.base = base
        else:
            self.base = base.med

        self.scan.base = self.base

        # Uses seted baseline toggle if possible or resets
        # if substitution due to different axes is not possible
        if self.sub_baseline_toggle.value and \
        all(_exp_data.index == self.scan.base.index):
            self.scan.sub_base(inplace=True)
        else:
            self.sub_baseline_toggle.value = False

    def update(self, *args, **kwargs):
        """Update the data on value change"""
        # Hackaround for not automatic valueupdate problem
        # when ffolder is changed on the fly
        try:
            fpath = os.path.join(self.ffolder, self.fpath_selector.value)
            fbase = os.path.join(self.ffolder, self.fbase_selector.value)
        except TypeError:
            fbase = self.ffolder
            fpath = self.ffolder
        if not os.path.isfile(fpath) or not os.path.isfile(fbase):
            warnings.warn('File does not exist. Skipping update\n%s\n%s'
                          % (fpath, fbase))
            return
        self.get(fpath, fbase )

    def update_scan(self, *args):
        """update scan related things here."""

        # there can be different indeces and then we dont want
        # baseline substitution we also need to make shure that
        # baseline toggle is only caled in case of this function has
        # been triggered by the coresponding function

        if isinstance(self.data, type(None)):
            warnings.warn("No data to update in %s" % self)
        elif isinstance(self.scan.base, type(None)):
            warnings.warn('No scan.base in %s' % self.scan)
        else:
            if args[0].get('owner') is self.sub_baseline_toggle:
                if all(self.data.index == self.scan.base.index):
                    if self.sub_baseline_toggle.value:
                        self.scan.sub_base(inplace=True)
                    else:
                        self.scan.add_base(inplace=True)

        self.update_data()

    def update_data(self, *args):
        """ update the viewd data itself """
        if isinstance(self.scan, TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        elif isinstance(self.scan, Scan):
            self.data = self.scan.med
        else:
            warnings.warn('No data to update in %s' % self)
            return

        if self.spec_selector.value != 'All':
            self.data = self.data[self.spec_selector.value]

    @property
    def widget_status(self):
        """Save widget config as ppWidget.json in ffolder"""
        def getattr_value_from_widget(*args):
            return getattr(getattr(self,  *args), "value")

        # Status of widget elements
        widget_status = []
        for widget_name in self._widgets:
            widget_status.append(getattr_value_from_widget(widget_name))
        widget_status = list(zip(self._widgets, widget_status))
        return widget_status

    def load_widget_status(self, widget_status):
        """load status of widget from widget_status

        Parameters
        ----------
        widget_status: dict
            Dict to load widget status from"""

        for widget_name, widget_value in widget_status:
            #print(widget_name, widget_value)
            widget = getattr(self, widget_name)
            widget.value = widget_value


class PumpProbeDataImporter(DataImporter):
    """ Very Similar to the DataImporter but the handling of spectra is
    slightly different. This one needs two """
    # List of widget names used to obtain widget status
    _widgets = (
        'fpath_selector', 'fbase_selector',
        'pump_selector',  'probe_selector', 'sub_baseline_toggle',
        'normalize_toggle'
    )

    def __init__(self, ffolder,
                 fpath_selector, fbase_selector, pp_delay_slider,
                 pump_selector, probe_selector, sub_baseline_toggle,
                 normalize_toggle, norm_widget):
        """
            fpath_seeelector:
                Selection widget for fpath

            fbase_selector:
                Selection widget for the baseline

            pp_delay_slider:
                Slider for Pump-Probe Timedelay

            pump_selector: Selection widget for the Pump

            probe_selection: Selection Widget for the Probe

            sub_baseline_toggle:
                Toggle button for substracting baseline

            norm_widget: widget_obj
                widget used to obtain norm_widget.data
        """

        # Init Properties
        self.ffolder = ffolder
        self.fpath_selector = fpath_selector
        self.fbase_selector = fbase_selector
        self.pp_delay_slider = pp_delay_slider
        self.pump_selector = pump_selector
        self.probe_selector = probe_selector
        self.sub_baseline_toggle = sub_baseline_toggle
        self.normalize_toggle = normalize_toggle
        self.norm_widget = norm_widget

    def __call__(self, title='', x_label='', y_label=''):
        """Create the widget by calling it """
        # Hackaround for not automatic valueupdate problem
        # when ffolder is changed on the fly
        from IPython.display import display
        from ipywidgets import HBox, VBox
        try:
            fpath = os.path.join(self.ffolder, self.fpath_selector.value)
            fbase = os.path.join(self.ffolder,  self.fbase_selector.value)
        except TypeError:
            fpath = self.ffolder
            fbase = self.ffolder
        if not os.path.isfile(fpath) or not os.path.isfile(fbase):
            warnings.warn('File does not exist. Skipping update\n%s\n%s'
                          % (fpath, fbase))
            self.fig = self.fig_init(title,x_label, y_label)
        else:
            # Init the Widget
            self.get(fpath, fbase)
            self.update_data()
            self.fig = self.fig_init(title,x_label, y_label)
            self.fig_update()

        # Init the observere
        self.observe()

        # Display the Widget
        display(
            HBox([self.fpath_selector, self.fbase_selector]),
            self.pp_delay_slider,
            HBox([self.pump_selector, self.probe_selector]),
            HBox([self.sub_baseline_toggle, self.normalize_toggle, self.fig.smooth_width]),
            VBox([self.fig, self.fig.pyplot, self.fig.tb_py])
        )

    def observe(self):
        """ Init the observers """
        self.fpath_selector.observe(self.update, 'value')
        self.fpath_selector.observe(self.update_data, 'value')
        self.fpath_selector.observe(self.fig_update, 'value')

        self.fbase_selector.observe(self.update, 'value')
        self.fbase_selector.observe(self.update_data, 'value')
        self.fbase_selector.observe(self.fig_update, 'value')

        self.pp_delay_slider.observe(self.update_data, 'value')
        self.pp_delay_slider.observe(self.fig_update, 'value')

        self.pump_selector.observe(self.update_data, 'value')
        self.pump_selector.observe(self.fig_update, 'value')

        self.probe_selector.observe(self.update_data, 'value')
        self.probe_selector.observe(self.fig_update, 'value')

        self.sub_baseline_toggle.observe(self.update_scan, 'value')
        self.sub_baseline_toggle.observe(self.fig_update, 'value')

        self.normalize_toggle.observe(self.update_scan, 'value')
        self.normalize_toggle.observe(self.fig_update, 'value')

    def get(self, fpath, fbase):
        """get data from HDD"""
        super().get(fpath, fbase)
        if isinstance(self.norm_widget.data, type(None)):
            warnings.warn("No valid normalization in %s", self.norm_widget)
            return
        self.scan.norm = self.norm_widget.data
        if self.normalize_toggle.value:
            self.scan.normalize(inplace=True)

    def update_data(self, *args):
        """ update the data viewd in the plot"""
        if isinstance(self.scan, TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        elif isinstance(self.scan, Scan):
            self.data = self.scan.med
        # In case of invalid call we just dot to anything
        else:
            warnings.warn('No data to plot in %s' % self)
            return

        self.scan.pump = self.pump_selector.value
        self.scan.probe = self.probe_selector.value
        self.scan.norm = self.norm_widget.data

    def update_scan(self, *args):
        """update the scan itself"""
        if isinstance(self.scan, type(None)):
            warnings.warn('No scan to normalize in %s' % self)
            return

        # Normalize was toggeled
        if args[0].get('owner') is self.normalize_toggle:
            if isinstance(self.scan.norm, type(None)):
                warnings.warn('No normalization data in %s', self.scan.norm)
            else:
                if self.normalize_toggle.value:
                    self.scan.normalize(inplace=True)
                else:
                    self.scan.un_normalize(inplace=True)
            self.update_data()
        # Sub Baseline was toggeled
        else:
            super().update_scan(*args)


######################################################################
#                         Using BQ-Plot                              #
######################################################################


class PPDelaySlider():
    def __init__(self, pp_delays=[0], data=None):
        """Widget with slider for pp_delays and smoothing of data

        Parameters
        ----------
        pp_delays : list
            Iterable of pp_delays to build the slider with

        data : DataFrame or Series
            data to use for the plot that is shown"""
        self._set_obj(pp_delays, data)
        self._init_widgets()
        self._init_widget_options()
        self._init_figure()
        self._init_observer()

    def __call__(self):
        from IPython.display import display
        display(self._container)
        self.fig

    def _set_obj(self, pp_delays, data):
        '''Set internal obj'''
        self.fig = None
        self.ax = None
        self.pp_delays = pp_delays
        self.data = data

    def __del__(self):
        self.w_smooth_s.close()
        self.w_pp_s.close()
        del self.fig
        del self.ax
        del self.w_pp_s
        del self.w_smooth_s
        del self.pp_delay
        del self.data

    def _init_widgets(self):
        import ipywidgets as wi

        self.w_pp_s = wi.SelectionSlider(
            continuous_update=False, description="pp_delay"
        )
        self.w_smooth_s = wi.IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=100
        )
        self.w_Autoscale = wi.ToggleButton(
            description="Autoscale",
            value=True,
        )
        # The container with all the widgets
        self._container = wi.HBox([self.w_Autoscale, self.w_pp_s, self.w_smooth_s])

    def _init_widget_options(self):
        self.w_pp_s.options = self.pp_delays
        self.w_pp_s.value = self.pp_delays[0]

    def _init_figure(self):
        if not self.fig:
            self.fig = plt.gcf()

        if not self.ax:
            self.ax = plt.gca()

        if isinstance(self.data, type(None)):
            if debug:
                print('Init empty Figure')
            return
        #Scan doesn't have multiindex and no pp_delays
        if not hasattr(self.data.index, 'levshape'):
            if debug:
                print('Has no levelshape')
            self.data.rolling(self.w_smooth_s.value).mean().plot(ax=self.ax)
            return

        try:
            self.data.ix[self.w_pp_s.value].rolling(
                self.w_smooth_s.value
            ).mean().plot(ax=self.ax)
        # Rolling doesnt work on duplicated axes
        except ValueError:
            if debug:
                print("In the init ValueError")
            self.data.ix[self.w_pp_s.value].plot(ax=self.ax)

    def _update_plot(self, new):
        if isinstance(self.data, type(None)):
            if debug:
                print("Updating figure with None data, Skipping ...")
            return

        # Scan doesnt have multiindex and no pp_delays
        if not hasattr(self.data.index, 'levshape'):
            data = self.data.rolling(self.w_smooth_s.value).mean()
            for line, spec in zip(self.ax.lines, self.data):
                # data = self.data[spec]
                line.set_ydata(data[spec].values)
                line.set_xdata(data[spec].index)
            return

        if isinstance(self.data, DataFrame):
            if any(self.data.columns.duplicated()):
                if debug:
                    print("Running as duplicated")
                for line, spec_line in zip(self.ax.lines, self.data.ix[self.w_pp_s.value].T.values):
                    line.set_ydata(spec_line)
            else:
                if debug:
                    print("Running as normal df")
                for line, spec in zip(self.ax.lines, self.data):
                    line.set_ydata(
                        self.data[spec].ix[self.w_pp_s.value].rolling(
                            self.w_smooth_s.value
                        ).mean().values
                    )

        elif isinstance(self.data, Series):
            if debug:
                print("Running as Series")
            self.ax.lines[0].set_ydata(
                self.data.ix[self.w_pp_s.value].rolling(
                    self.w_smooth_s.value
                ).mean().values
            )
        else:
            raise NotImplementedError("Can't handle datatype of %f"
                                      % type(self.data))
        self.ax.figure.canvas.draw()

    def _update_axes(self, new):
        if self.w_Autoscale.value:
            self.ax.relim()
            self.ax.autoscale_view()

    def _init_observer(self):
        """init all observers"""
        self.w_pp_s.observe(self._update_plot, "value")
        self.w_pp_s.observe(self._update_axes, "value")
        self.w_smooth_s.observe(self._update_plot, "value")
        self.w_smooth_s.observe(self._update_axes, "value")
        self.w_Autoscale.observe(self._update_axes, "value")

    def _unobserve(self):
        """unobserve all observers"""
        self.w_pp_s.unobserve_all()
        self.w_smooth_s.unobserve_all()
        self.w_Autoscale.unobserve_all()

class Importer(PPDelaySlider):
    def __init__(self, ffolder='/'):
        """
        Parameters
        ----------
        ffolder : str
            Folder to list data in.

        """
        import ipywidgets as wi

        self.ffolder = ffolder
        files = sorted(glob(ffolder + '/*.dat'))
        files = [ x for x in files if "AVG" not in x ]
        fnames = [os.path.split(ffile)[1] for ffile in files]
        self.w_files = wi.Select(
            description='File',
            options = fnames
            )
        if self.w_files.options != []:
            self.w_files.value = self.w_files.options[0]
        self.w_files.layout.width = '100%'

        pp_delays, data = self._get_data()
        super().__init__(pp_delays, data.med)
        self._container = wi.VBox([self.w_files, self._container])

    def _get_data(self):
        if not self.w_files.value:
            # open default dummy file
            return [0], 
        data = read_auto(self.ffolder + '/'  + self.w_files.value)
        pp_delays = [0]
        # because scans dot have pp_delays
        if hasattr(data, 'pp_delays'):
            pp_delays = data.pp_delays
        return pp_delays, data

    def _update_file(self, new):
        if self.fig:
            self.fig.clear()
        pp_delays, data = self._get_data()
        self._set_obj(pp_delays, data.med)
        super()._unobserve()
        self._init_widget_options()
        super()._init_observer()
        self._init_figure()

    def _init_observer(self):
        self.w_files.observe(self._update_file, 'value')
        super()._init_observer()

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
