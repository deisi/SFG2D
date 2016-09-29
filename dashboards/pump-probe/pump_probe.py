import os
import json
import warnings
import ipywidgets as ipyw
import matplotlib.pyplot as plt
import sfg2d

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from glob import glob

debug = 0

class FileHandler(FileSystemEventHandler):
    ppWidget = None

    def __init__(self, ffolder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffolder = ffolder
        self.ffiles = glob(self.ffolder + '/*.dat')
        self.ffiles = [ x for x in self.ffiles if "AVG" not in x ]
        self.fnames = [os.path.split(ffile)[1] for ffile in self.ffiles]
    
    def on_any_event(self, event):
        if debug:
            print('got event', event)
        self.ffiles = glob(self.ffolder + '/*.dat')
        self.ffiles = [ x for x in self.ffiles if "AVG" not in x ]
        self.fnames = sorted(
            [os.path.split(ffile)[1] for ffile in self.ffiles]
        )
        if not self.ppWidget:
            return
        self.ppWidget.ir_fpath.options = self.fnames

class PumpProbeWidget():
    w_ir = None # Widget for IR spectrum
    w_pump = None # Widget for pump Spectrum 
    w_pump_probe = None # Widget for pump_probe spectrum
    # List of attached widgets to save and load
    _widgets = ('w_ir', 'w_pump', 'w_pump_probe')
    
    def __init__(self, fnames):
        """Init PumpProbeWidget

        Parameters
        ----------
        """
        self.ir_fpath = ipyw.Select(options=fnames, description='IR Profile')
        self.ir_fbase = ipyw.Select(description='IR Base')
        self.ir_spec = ipyw.Select(
            options = ["All", "spec_0", "spec_1", "spec_2"], 
            description="Spectrum", value="All"
        )
        self.ir_sub_base = ipyw.ToggleButton(
            description='Sub Baseline',
            value=False
        )
        self.ir_ppdelay = ipyw.SelectionSlider(continuous_update=False)

        self.pump_fpath = ipyw.Select(description='Pump SFG')
        self.pump_fbase = ipyw.Select(description='Pump Base')
        self.pump_spec = ipyw.Select(
            options = ["All", "spec_0", "spec_1", "spec_2"], 
            description="Spectrum", value="spec_1"
        )
        self.pump_ppdelay = ipyw.SelectionSlider(continuous_update=False)
        self.pump_sub_base = ipyw.ToggleButton(
            description='Sub Baseline',
            value=False
        )

        self.ts0_fpath = ipyw.Select(description='Pump Probe')
        self.ts0_pumped = ipyw.Dropdown(
            options = ["spec_0", "spec_1", "spec_2"], 
            description="pumped", value="spec_0"
        )
        self.ts0_probed = ipyw.Dropdown(
            options = ["spec_0", "spec_1", "spec_2"], 
            description="probed", value="spec_1"
        )
        self.ts0_sub_base = ipyw.ToggleButton(
            description='Sub Baseline', value=False
        )
        self.ts0_normalize = ipyw.ToggleButton(
            description='Normalize', value=False
        )
        self.ts0_ppdelay = ipyw.SelectionSlider(continuous_update=False)
        self.ts0_ppdelay_childs = [ ipyw.SelectionSlider(continuous_update=False)
                          for i in range(4)
        ]

        self.fbase = ipyw.Select(description='Baseline')


        self._l_fpath = {}
        self._l_ts0_ppdelay_childs = []

    def linkTraitlets(self):
        """ links the traitlets in the gui """
        import traitlets

        # Database is for each data selector the same. Thus
        # the options can all be linked to self.ir_fpath.options
        for _name, _w in (('ir_fbase',self.ir_fbase), 
                          ('pump_fpath',self.pump_fpath),
                          ('pump_fbase', self.pump_fbase), 
                          ('ts0_fpath', self.ts0_fpath),
                          ('ts0_fbase', self.fbase)):
             self._l_fpath[_name] = (
                 traitlets.dlink((self.ir_fpath, 'options'), (_w, 'options'))
             )

        # The pp_delay sliders are all the same
        for _w in self.ts0_ppdelay_childs:
            self._l_ts0_ppdelay_childs.append(
                traitlets.dlink((self.ts0_ppdelay, 'options'),
                                (_w, 'options'))
            )

    def unlinkTraitlets(self):
        """ unlink all traitlets """
        for key in self._l_fpath:
            link = self._l_fpath[key]
            link.unlink()

        for link in self._l_ts0_ppdelay_childs:
            link.unlink()

    def save(self, ffolder):
        """Save widget config as ppWidget.json in ffolder"""

        ffolder = os.path.normpath(ffolder)
        # First data must be prepared
        widget_dict = {}
        for widget_name in self._widgets:
            try:
                widget_dict[widget_name] = getattr(getattr(self, widget_name), 'widget_status')
            except AttributeError:
                # when a widget is not jet registered
                pass

        # The actual save
        with open(ffolder + '/ppWidget.json', 'w') as outfile:
            json.dump(widget_dict, outfile)
            outfile.close()
            
    def load(self, ffolder):
        ffolder = os.path.normpath(ffolder)
        try:
            with open(ffolder + '/ppWidget.json', 'r') as infile:
                data = json.load(infile)
                infile.close()
                
                for widget_name in self._widgets:
                    try :
                        widget = getattr(self, widget_name)
                        widget_status = data[widget_name]
                        widget.load_widget_status(widget_status)

                    # Because an individual widget_status can be
                    # incomplte
                    except KeyError:
                        pass
                    
        except FileNotFoundError:
            warnings.warn('No ppWidget.json file in %s' % ffolder)
            pass    

        
def gen_PumpProbeWidget(ffolder):
    ffolder = os.path.normpath(os.path.abspath(ffolder))
    
    # Observer to monitor changes in ffolder
    observer = Observer()
    event_handler = FileHandler(ffolder)

    # Widget for the PumpProbe Dashboard
    ppWidget = PumpProbeWidget(event_handler.fnames)

    # link eventhandler and widget, so widget is informed
    # by event_handler in case of file changes
    event_handler.ppWidget = ppWidget

    # Start file observer to monitor
    observer.schedule(event_handler, ffolder, recursive=False)
    if os.path.isdir(ffolder):
        observer.start()

    # Link Traitlets to have the same files available in all widgets
    ppWidget.linkTraitlets()

    # Stup individual widgets
    ppWidget.w_ir = sfg2d.widgets.DataImporter(ffolder, ppWidget.ir_fpath, 
                               ppWidget.fbase, ppWidget.ir_ppdelay, 
                               ppWidget.ir_spec, ppWidget.ir_sub_base)
    ppWidget.w_pump = sfg2d.widgets.DataImporter(ffolder, ppWidget.pump_fpath, 
                               ppWidget.pump_fbase, ppWidget.pump_ppdelay, 
                               ppWidget.pump_spec, ppWidget.pump_sub_base)
    ppWidget.w_pump_probe = sfg2d.widgets.PumpProbeDataImporter(
        ffolder, ppWidget.ts0_fpath, 
        ppWidget.fbase, ppWidget.ts0_ppdelay, 
        ppWidget.ts0_pumped, ppWidget.ts0_probed, ppWidget.pump_sub_base,
        ppWidget.ts0_normalize, norm_widget = ppWidget.w_ir
    )
    ppWidget.load(ffolder)
    return ppWidget, observer, event_handler

def update_ffolder(ffolder, observer, ppWidget):
    """update eventhandler and widget if folder is changed 

    Parameters
    ----------
    observer:

    ppWidget:

    ffolder:

    """
    ffolder = os.path.normpath(os.path.abspath(ffolder))

    # Because the gui should do nothing if the new file is not valid.
    if not os.path.isdir(ffolder):
        return ppWidget, observer

    # The observer must be properly stoped
    # before it can be restarted again.
    if observer.isAlive():
        observer.unschedule_all()
        observer.stop()
        observer.join()
    
    observer = Observer()
    event_handler = FileHandler(ffolder)

    ppWidget.w_ir.ffolder = ffolder
    ppWidget.w_pump.ffolder = ffolder
    ppWidget.w_pump_probe.ffolder = ffolder

    # Allow event_handler to update ppWidget fpath properties
    # so changes in the new folde can be monitored
    event_handler.ppWidget = ppWidget

    # Otherwise errors are thrown because observed widgets get updated
    # but data is not available jet.
    for sub_widget in (ppWidget.w_ir, ppWidget.w_pump, ppWidget.w_pump_probe):
        sub_widget.unobserve()
    #ppWidget.unlinkTraitlets()
    ppWidget.ir_fpath.options = event_handler.fnames
    for sub_widget in (ppWidget.w_ir, ppWidget.w_pump, ppWidget.w_pump_probe):
        sub_widget.observe()
    ppWidget.linkTraitlets()
    event_handler.ppWidget = ppWidget

    # Start file observer to monitor filechanges in new folder
    observer.schedule(event_handler, ffolder, recursive=False)
    observer.start()

    #ppWidget.load(ffolder)

    return ppWidget, observer, event_handler
