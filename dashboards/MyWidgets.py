from IPython.display import display
import SFG2D

from bqplot import (
    LogScale, LinearScale, OrdinalColorScale, ColorAxis,
    Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip
)


class DataImporter():
    data = None # Data presented by the widget
    scan = None # Full data used by the widget
    base = None # Full Baseline data

    def __init__(self, ffolder, fpath_selector,
                 fbase_selector,  pp_delay_slider,
                 spec_selector, sub_baseline_toggle):
        '''
        scan: SFG2D.scan obj
            This is the containre obj for all the data, to interface with the
            outside wold.

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

    def __call__(self, title='',x_label='', y_label=''):

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

        self.sub_baseline_toggle.observe(self.sub_baseline_update, 'value')
        self.sub_baseline_toggle.observe(self.fig_update, 'value')

        # Display the Widget
        display(
            self.fpath_selector, self.fbase_selector, self.pp_delay_slider,
            self.spec_selector, self.sub_baseline_toggle, self.fig
        )

    def get(self, fpath, fbase):
        """Get Data from HDD

        fpath: path to data

        fbase: path to background data"""
        data = SFG2D.io.veronica.read_auto(fpath)
        self.scan = data
        if isinstance(data, SFG2D.core.scan.TimeScan):
            self.pp_delay_slider.options = list(data.pp_delays)
#            self.data = self.scan.med.ix[self.pp_delay_slider.value]
        else:
#            self.data = self.scan.med
            self.pp_delay_slider.options = [0]
            self.pp_delay_slider.value = 0
#
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
#
#        if self.spec_selector.value is not 'All':
#            self.data = self.data[self.spec_selector.value]
#
        if self.sub_baseline_toggle.value:
            self.scan.sub_base(inplace=True)
#            # Needed to allow changing data while a spec is selected
#            if self.spec_selector.value is not 'All':
#                self.data -= self.scan.base[self.spec_selector.value]
#            else:
#                self.data -= self.scan.base


    def update(self, *args, **kwargs):
        """Update the data on value change"""
        self.get(self.ffolder + self.fpath_selector.value,
                 self.ffolder + self.fbase_selector.value)
        
    def fig_init(self, title='',x_label='', y_label=''):
        """Init an empty bqplot figure"""
        x_sc = LinearScale()
        y_sc = LinearScale()

        line = Lines(scales={'x':x_sc, 'y':y_sc})

        ax_x = Axis(scale=x_sc,
                    label=x_label)
        ax_y = Axis(scale=y_sc, 
                    label=y_label, orientation='vertical')

        fig = Figure(marks=[line], axes=[ax_x, ax_y], 
                     title=title)
        return fig

    def fig_update(self, *args, **kwargs):
        """Update bqplot figure with new data"""
        self.fig.marks[0].x = self.data.index
        self.fig.marks[0].y = self.data.transpose()

    def sub_baseline_update(self, *args):

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
            
        if self.spec_selector.value is not "All":
            self.data = self.data[self.spec_selector.value]
