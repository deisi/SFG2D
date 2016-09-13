import SFG2D
from IPython.display import display
from bqplot import LinearScale, Axis, Lines, Figure, Toolbar, PanZoom
from ipywidgets import VBox, HBox, ToggleButton, BoundedIntText
from traitlets import TraitError
import json
import warnings
import os


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
        data = SFG2D.io.veronica.read_auto(fpath)
        self.scan = data
        if isinstance(data, SFG2D.core.scan.TimeScan):
            self.pp_delay_slider.options = list(data.pp_delays)
            _exp_data = data.med.ix[list(data.pp_delays)[0]]
        else:
            self.pp_delay_slider.options = [0]
            self.pp_delay_slider.value = 0
            _exp_data = data.med

        base = SFG2D.io.veronica.read_auto(fbase)
        if isinstance(base, SFG2D.core.scan.TimeScan) and not isinstance(data,  SFG2D.core.scan.TimeScan):
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
        if isinstance(self.scan, SFG2D.core.scan.TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        elif isinstance(self.scan, SFG2D.core.scan.Scan):
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
        if isinstance(self.scan, SFG2D.core.scan.TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        elif isinstance(self.scan, SFG2D.core.scan.Scan):
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
