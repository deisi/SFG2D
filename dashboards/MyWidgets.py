import SFG2D
from IPython.display import display
from bqplot import LinearScale, Axis, Lines, Figure, Toolbar, PanZoom
from ipywidgets import VBox, HBox
from traitlets import TraitError
import json
import warnings


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
        pz = PanZoom(scales={'y': [y_sc]})

        fig = Figure(marks=[line], axes=[ax_x, ax_y], 
                     title=title)

        tb = Toolbar(figure=fig)
        tb._panzoom = pz

        fig.pyplot = tb
        #fig.interaction = pz
        
        return fig

    def fig_update(self, *args, **kwargs):
        """Update bqplot figure with new data"""
        self.fig.marks[0].x = self.data.index
        self.fig.marks[0].y = self.data.transpose()


class DataImporter(MyBqPlot):
    data = None # Data presented by the widget
    scan = None # Full data used by the widget
    base = None # Full Baseline data
    # List of widget names used to obtain widget status
    _widgets = (
        'fpath_selector', 'fbase_selector', 'pp_delay_slider',
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

        # Init the Widget
        self.get(self.ffolder + self.fpath_selector.value,
                 self.ffolder + self.fbase_selector.value)
        self.update_data()
        self.fig = self.fig_init(title,x_label, y_label)
        self.fig_update()
        
        # Link the observers
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

        # Display the Widget
        display(
            HBox([self.fpath_selector, self.fbase_selector]),
            HBox([self.pp_delay_slider, self.spec_selector])
            , self.sub_baseline_toggle,
            VBox([self.fig, self.fig.pyplot])
        )

    def get(self, fpath, fbase):
        """Get Data from HDD

        fpath: path to data

        fbase: path to background data"""
        data = SFG2D.io.veronica.read_auto(fpath)
        self.scan = data
        if isinstance(data, SFG2D.core.scan.TimeScan):
            self.pp_delay_slider.options = list(data.pp_delays)
        else:
            self.pp_delay_slider.options = [0]
            self.pp_delay_slider.value = 0

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

        if self.sub_baseline_toggle.value:
            self.scan.sub_base(inplace=True)

    def update(self, *args, **kwargs):
        """Update the data on value change"""
        self.get(self.ffolder + self.fpath_selector.value,
                 self.ffolder + self.fbase_selector.value)
        
    def update_scan(self, *args):

        if self.sub_baseline_toggle.value:
            self.scan.sub_base(inplace=True)
        else:
            self.scan.add_base(inplace=True)

        self.update_data()

    def update_data(self, *args):
        if isinstance(self.scan, SFG2D.core.scan.TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        else:
            self.data = self.scan.med
            
        if self.spec_selector.value != 'All':
            self.data = self.data[self.spec_selector.value]
                        
    @property
    def widget_status(self):
        """Save widget config as ppWidget.json in ffolder"""
        def getattr_value_from_widget(*args):
            return getattr(getattr(self,  *args), "value")

        # Status of widget elements
        widget_status = {}
        for widget_name in self._widgets:
            widget_status[widget_name] = getattr_value_from_widget(widget_name)
        return widget_status

    def load_widget_status(self, widget_status):
        """load status of widget from widget_status

        Parameters
        ----------
        widget_status: dict
            Dict to load widget status from"""

        for key in widget_status:
            widget = getattr(self, key)
            try:
                widget.value = widget_status[key]
            # happens if attributes are loaded in the wrong order
            except TraitError:
                warnings.warn("Error loading widget status at %s" % key)
            

class PumpProbeDataImporter(DataImporter):
    """ Very Similar to the DataImporter but the handling of spectra is
    slightly different. This one needs two """
    # List of widget names used to obtain widget status
    _widgets = (
        'fpath_selector', 'fbase_selector', 'pp_delay_slider',
        'pump_selector',  'probe_selector', 'sub_baseline_toggle',
        'normalize_toggle'
    )

    def __init__(self, ffolder, 
                 fpath_selector, fbase_selector, pp_delay_slider,
                 pump_selector, probe_selector, sub_baseline_toggle,
                 normalize_toggle, norm_scan=None):
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

            norm_scan: Scan obj 
                Scan used for normalization
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
        self.norm = norm_scan

    def __call__(self, title='', x_label='', y_label=''):

        # Init the Widget
        self.get(self.ffolder + self.fpath_selector.value,
                 self.ffolder + self.fbase_selector.value)
        self.update_data()
        self.fig = self.fig_init(title,x_label, y_label)
        self.fig_update()
        
        # Link the observers
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

        # Display the Widget
        display(
            HBox([self.fpath_selector, self.fbase_selector]),
            self.pp_delay_slider,
            HBox([self.pump_selector, self.probe_selector]),
            HBox([self.sub_baseline_toggle, self.normalize_toggle]),
            VBox([self.fig, self.fig.pyplot])
        )

    def get(self, fpath, fbase):
        super().get(fpath, fbase)
        self.scan.norm = self.norm
        if self.normalize_toggle.value:
            self.scan.normalize(inplace=True)
    
    def update_data(self, *args):
        if isinstance(self.scan, SFG2D.core.scan.TimeScan):
            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        else:
            self.data = self.scan.med

        self.scan.pump = self.pump_selector.value
        self.scan.probe = self.probe_selector.value
        self.scan.norm = self.norm

    def update_scan(self, *args):

        # Normalize was toggeled
        if args[0]['owner'] is self.normalize_toggle:
            if self.normalize_toggle.value:
                self.scan.normalize(inplace=True)
            else:
                self.scan.un_normalize(inplace=True)
            self.update_data()
        # Sub Baseline was toggeled
        else:
            super().update_scan(*args)
