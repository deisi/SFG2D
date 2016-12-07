from IPython.display import display
from bqplot import LinearScale, Axis, Lines, Figure, Toolbar, PanZoom
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from ipywidgets import VBox, HBox, ToggleButton, BoundedIntText, SelectionSlider, IntSlider, Select, IntRangeSlider, FloatText, Dropdown, Text
from traitlets import TraitError
from glob import glob
import json
import warnings
import os
import numpy as np

from .io.veronica import read_auto
from .io.allYouCanEat import AllYouCanEat, x_pixel_index, y_pixel_index, spec_index, frame_axis_index, pp_index
from .core.scan import Scan, TimeScan

debug = 1

class MyBqPlot():
    """Class for Bqolot setup. """

    def fig_init(self, title='',x_label='', y_label=''):
        """Init an empty bqplot figure"""
        x_sc = LinearScale()
        y_sc = LinearScale()

        line = Lines(scales={'x':x_sc, 'y':y_sc})

        ax_x = Axis(scale=x_sc,
                    label=x_label)
        ax_y = Axis(scale=y_sc, 
                    label=y_label, orientation='vertical')

        # Zoom only y-scale
        def py_update(new):
            if tb_py.value:
                py = PanZoom(scales={'y': [y_sc]})
                fig.interaction = py
            else:
                fig.interaction.close()
        tb_py = ToggleButton(description="Zoom-Y")
        tb_py.observe(py_update, "value")

        # Smooth data here because it only effects the way the 
        # data is shown, it has not deeper physical meaning
        # and doesnt interfere with the data containing classes
        smooth_width = BoundedIntText(
            value=0, min=0, max=100,
            description = 'Smooth',
            tooltip = 'Size of the windows used for rolling median to smooth data'
            )
        smooth_width.observe(self.fig_update, "value")

        fig = Figure(marks=[line], axes=[ax_x, ax_y], 
                     title=title)

        tb = Toolbar(figure=fig)

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
#                         A new Approach                             #
######################################################################
class GuiABS():
    def __init__(self, data=AllYouCanEat(), fig=None, ax=None):
        self.data = data
        self.fig = fig
        self.ax = ax
        self.widget_container = [] # List of widgets to display

    def __call__(self):
        display(self.widget_container)
        self.fig

    def _init__widgets(self):
        """Init all widgets."""
        pass

    def _set_widget_options(self):
        """Set all widget options."""
        pass

    def _init_figure(self):
        """Init initial figure."""
        if not self.fig:
            self.fig = plt.gcf()

        if not self.ax:
            self.ax = plt.gca()

    def _update_figure(self):
        """Update figure with changes"""
        pass

    def _init_observer(self):
        pass

    def _unobserve(self):
        pass

class AllYouCanPlot(GuiABS):
    def __init__(self, central_wl=None, vis_wl=810, **kwargs):
        super().__init__(**kwargs)
        self._central_wl = central_wl
        self.vis_wl = vis_wl

    def _init__widgets(self):
        """Init all widgets."""
        self.w_pp_s = SelectionSlider(
            continuous_update=False, description="pp_delay"
        )
        self.w_smooth_s = IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=100
        )
        self.w_Autoscale = ToggleButton(
            description="Autoscale",
            value=True,
        )
        self.w_frame = SelectionSlider(
            continuous_update=False, description="frame"
        )
        self.w_y_pixel_range = IntRangeSlider(
            continuous_update=False, description="y_pixels"
        )
        self.w_x_pixel_range = IntRangeSlider(
            continuous_update=False, description="x_pixels"
        )

        self.w_central_wl = FloatText(
            description='Central Wl', value=self.central_wl, width='120px'
        )
        self.w_calib = Dropdown(
            description='Calibration', options=['pixel', 'nm', 'wavenumber'], width='60px'
        )
        self.w_vis_wl = FloatText(
            description = 'Vis Wl', value = self.vis_wl, width = '120px'
        )
        self.w_folder = Text(description="Folder")
        self.w_file = Select(descripion='Files')

        # The container with all the widgets
        self.widget_container = VBox([
                HBox([self.w_folder, self.w_file,]),
                HBox([self.w_Autoscale, self.w_pp_s, self.w_smooth_s]),
                HBox([self.w_frame, self.w_y_pixel_range, self.w_x_pixel_range]),
                self.w_vis_wl, self.w_central_wl, self.w_calib
            ])

    def _set_widget_options(self):
        """Set all widget options."""
        self.w_pp_s.options = self.data.pp_delays.tolist()
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
        self.w_y_pixel_range.value = self.w_y_pixel_range.min, self.w_y_pixel_range.max
        if self.data.data.shape[y_pixel_index] == 1:
            self.w_y_pixel_range.disabled = True
        else:
            self.w_y_pixel_range.disabled = False
        self.w_x_pixel_range.max = self.data.data.shape[x_pixel_index]
        self.w_x_pixel_range.value = self.w_x_pixel_range.min, self.w_x_pixel_range.max

    @property
    def central_wl(self):
        central_wl = 674
        if self._central_wl == None:
            if self.data.metadata.get('central_wl') != 0:
                central_wl = self.data.metadata.get('central_wl')
        else:
            central_wl = self._central_wl
        return central_wl

    @property
    def y(self):
        """prepare y data to plot."""
        # Needed to account for overlapping of the range sliders.
        # y_range = self.w_y_pixel_range.value
        # if y_range[0] == y_range[1]:
        #     if self.data.data.shape[y_pixel_index] > y_range[0]:
        #         y_range = y_range[0], y_range[0] + 1
        #     else:
        #         y_range = y_range[1] - 1, y_range[1]
        y_slice = _slider_range_to_slice(self.w_y_pixel_range.value,
                                         self.data.data.shape[y_pixel_index])
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         self.data.data.shape[x_pixel_index])
        pp_delay_index = np.where(self.w_pp_s.value == self.data.pp_delays)[0][0]
        y = self.data.data[
            pp_delay_index,
            self.w_frame.value,
            y_slice,
            x_slice
        ].T
        return y

    @property
    def x(self):
        x_slice = _slider_range_to_slice(self.w_x_pixel_range.value,
                                         self.data.data.shape[x_pixel_index])
        if self.w_calib.value=='pixel':
            x = np.arange(self.data.data.shape[x_pixel_index])
        if self.w_calib.value=='nm':
            x = self.data.wavelength
        if self.w_calib.value=='wavenumber':
            x = self.data.wavenumber
        return x[x_slice]

    def _update_figure(self):
        """Init initial figure."""
        super()._init_figure()
        #self.fig.clf
        self.ax.clear()
        self._lines = plt.plot(self.x, self.y)
        self.ax.set_xlabel(self.w_calib.value)

    def _init_observer(self):
        self.w_folder.on_submit(self._on_folder_submit)
        self.w_file.observe(self._update_data, 'value')
        self.w_file.observe(self._update_figure_callback, 'value')
        self.w_calib.observe(self._update_figure_callback, 'value')
        self.w_frame.observe(self._update_figure_callback, 'value')
        self.w_pp_s.observe(self._update_figure_callback, 'value')
        self.w_y_pixel_range.observe(self._update_figure_callback, 'value')
        self.w_x_pixel_range.observe(self._update_figure_callback, 'value')

    def _unobserve(self):
        pass
        #TODO find out if I can use the widget_container for this

    def _update_figure_callback(self, new):
        self._update_figure()

    def _on_folder_submit(self, new):
        fnames = np.sort(glob(os.path.normcase(self.w_folder.value + '/*' )))
        # Only .dat and .spe in known
        mask = [a or b for a, b in zip([".dat" in s for s in fnames],  [".spe" in s for s in fnames])]
        fnames = fnames[np.where(mask)]
        # Remove AVG
        fnames = fnames[np.where(["AVG" not in s for s in fnames])]
        fnames = [os.path.split(elm)[1] for elm in fnames]

        if debug:
            print("_on_filder_submit_called")
        if debug > 1:
            print("fnames:", fnames)

        self.w_file.options = fnames

    def _update_data(self, new):
        fname = self.w_folder.value + "/" + self.w_file.value
        self.data = AllYouCanEat(fname)
        if ".spe" in self.w_file.value:
            self.w_central_wl.disabled = True
        else:
            self.w_central_wl.disabled = False


        self._set_widget_options()


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
        self.w_pp_s = SelectionSlider(
            continuous_update=False, description="pp_delay"
        )
        self.w_smooth_s = IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=100
        )
        self.w_Autoscale = ToggleButton(
            description="Autoscale",
            value=True,
        )
        # The container with all the widgets
        self._container = HBox([self.w_Autoscale, self.w_pp_s, self.w_smooth_s])

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
        self.ffolder = ffolder
        files = sorted(glob(ffolder + '/*.dat'))
        files = [ x for x in files if "AVG" not in x ]
        fnames = [os.path.split(ffile)[1] for ffile in files]
        self.w_files = Select(
            description='File',
            options = fnames
            )
        if self.w_files.options != []:
            self.w_files.value = self.w_files.options[0]
        self.w_files.layout.width = '100%'

        pp_delays, data = self._get_data()
        super().__init__(pp_delays, data.med)
        self._container = VBox([self.w_files, self._container])

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

def _slider_range_to_slice(value_tuple, max):
    """Transform a tuple into a slice accounting for overlapping"""
    range = value_tuple
    if range[0] == range[1]:
        if range[0] < max:
            range = range[0], range[0] + 1
        else:
            range = range[1] - 1, range[1]

    return slice(*range)
