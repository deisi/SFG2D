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
        # This allows the data to be used for normalization from start on
        self.data.data += 1
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
        y = self.data.data[:, :, y_slice, x_slice].copy()
        if self.w_sub_base.value:
            y -= np.ones_like(y) * self.y_base

        if self.w_frame_median.value:
            # TODO Maybe merge this with y property. They are redundant to some
            # extend.
            y = np.median(self.data.data, frame_axis_index).sum(x_pixel_index)
        else:
            y = y[:, self.w_frame.value, :, :].sum(x_pixel_index)

        return y

    @property
    def sum_x(self):
        """x data of the summed plot."""
        return self.data.pp_delays

    def _on_xlim_changed(self, new=None):
        self._ax_xlim = self.axes[0].get_xlim()

    def _on_ylim_changed(self, new=None):
        self._ax_ylim = self.axes[0].get_ylim()


#class Normalized(WidgetBase):
#@    def __init__(self, ir=AllYouCanEat().data, **kwargs):


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
