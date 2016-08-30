import os
import ipywidgets as ipyw

#from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from glob import glob



class PumpProbeDashboardData():
    def __init__(self, ffolder = '', ):
        """ Container class for the data needed by the widget """
        
        self.ffolder = ffolder
        self._ffiles = []
        self._fnames = []

        self.ir = None
        self.pump = None
        self.base = None
        self.ts0 = None
        self.ts0u = None

    @property
    def ffiles(self):
        if self._ffiles == [] and self.ffolder is not '':
           self._ffiles = glob(self.ffolder + '/*.dat')
           self._ffiles = [ x for x in self._ffiles if "AVG" not in x ]
        return self._ffiles

    @ffiles.setter
    def ffiles(self, value):
        self._ffiles = value

    @property
    def fnames(self):
        if self._fnames == [] and self.ffiles != []:
            self._fnames = [os.path.split(ffile)[1] for ffile in self._ffiles]
        return self._fnames

    @fnames.setter
    def fnames(self, value):
        self._fnames = value


# Watchdog to monitor ppdData.ffolder
class MyHandler(FileSystemEventHandler):
    ppdData = None
    ppWidget = None
    
    def on_modified(self, event):
        #print('jepjep')
        self.ppdData.ffiles = glob(self.ppdData.ffolder + '/*.dat')
        self.ppdData.ffiles = [ x for x in self.ppdData.ffiles if "AVG" not in x ]
        self.ppdData.fnames = [os.path.split(ffile)[1] for ffile in self.ppdData.ffiles]
        self.ppWidget.ir_fpath.options = self.ppdData.fnames


class PumpProbeWidget():
    def __init__(self, ppdData):
        """Init PumpProbeWidget

        Parameters
        ----------
        ppdData : PumpProbeDasboardData obj
            The object that holds the actual data and
            the information abot the data."""
        self.ir_fpath = ipyw.Select(options=ppdData.fnames, description='IR Profile')
        self.ir_fbase = ipyw.Select(description='IR Base')
        self.ir_spec = ipyw.Select(
            options = ["All", "spec_0", "spec_1", "spec_2"], 
            description="Spectrum", value="spec_1"
        )
        self.ir_sub_base = ipyw.ToggleButton(
            description='Sub Baseline',
            value=False
        )

        self.pump_fpath = ipyw.Select(description='Pump SFG')
        self.pump_fbase = ipyw.Select(description='Pump Base')
        self.pump_spec = ipyw.Select(
            options = ["All", "spec_0", "spec_1", "spec_2"], 
            description="Spectrum", value="spec_1"
        )
        self.pump_ppdelay = ipyw.SelectionSlider(continuous_update=False)

        self.ts0_fpath = ipyw.Select(description='Pump Probe')
        self.ts0_pumped = ipyw.Dropdown(
            options = ["spec_0", "spec_1", "spec_2"], 
            description="pumped", value="spec_0"
        )
        self.ts0_probed = ipyw.Dropdown(
            options = ["spec_0", "spec_1", "spec_2"], 
            description="probed", value="spec_1"
        )
        self.ts0_sub_base = ipyw.ToggleButton(description='Sub Baseline', value=False)
        self.ts0_normalize = ipyw.ToggleButton(description='Normalize', value=False)
        self.ts0_ppdelay = ipyw.SelectionSlider(continuous_update=False)
        self.ts0_ppdelay_childs = [ ipyw.SelectionSlider(continuous_update=False)
                          for i in range(4)
        ]

        self.fbase = ipyw.Select(description='Baseline')

        self._l_fpath = {}
        self._l_ts0_ppdelay_childs = []

    def setupObservers(self):
        """ Sets up the observers of the widget """
        def pump_plot_update(change):
            pump_plot(self.pump_spec.value, self.pump_ppdelay.value)  

        def pump_probe_plot_update(change):
            pump_probe_plot(self.ts0_ppdelay.value)

        def ts0_pumped_update(change):
            ts0._pumped = self.ts0_pumped.value
            if isinstance(ts0._df.get("bleach"), pd.core.series.Series):
                ts0.df.drop('bleach', axis=1, inplace=True)

        def ts0_proped_update(change):
            ts0._probed = self.ts0_probed.value
            if isinstance(ts0._df.get("bleach"), pd.core.series.Series):
                ts0.df.drop('bleach', axis=1, inplace=True)

        self.pump_fpath.observe(pump_plot_update, names="value")
        self.ts0_fpath.observe(pump_probe_plot_update, names="value")
        self.ts0_pumped.observe(ts0_pumped_update, names="value")
        self.ts0_probed.observe(ts0_proped_update, names="value")

    def linkTraitlets(self):
        """ links the traitlets in the gui """
        import traitlets

        # Database is for each data selector the same. Thus
        # the options can all be linked to self.ir_fpath.options
        for _name, _w in (('ir_fbase',self.ir_fbase), 
                          ('pump_fpath',self.pump_fpath), ('pump_fbase', self.pump_fbase), 
                          ('ts0_fpath', self.ts0_fpath), ('ts0_fbase', self.fbase)):
             self._l_fpath[_name] = (traitlets.dlink((self.ir_fpath, 'options'), (_w, 'options')))

        # The pp_delay sliders are all the same
        for _w in self.ts0_ppdelay_childs:
            self._l_ts0_ppdelay_childs.append(
                traitlets.dlink((self.ts0_ppdelay, 'options'),(_w, 'options'))
            )



