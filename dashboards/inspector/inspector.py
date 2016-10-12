from os import path
from IPython.display import display
from ipywidgets import Text
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from glob import glob

import sfg2d
import pandas as pd

debug = 0

class FileHandler(FileSystemEventHandler):
    def __init__(self, ffolder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffolder = ffolder
        self.ffiles = glob(self.ffolder + '/*.dat')
        self.ffiles = [ x for x in self.ffiles if "AVG" not in x ]
        self.fnames = [path.split(ffile)[1] for ffile in self.ffiles]
    
    def on_any_event(self, event):
        if debug:
            print('got event', event)
        self.ffiles = glob(self.ffolder + '/*.dat')
        self.ffiles = [ x for x in self.ffiles if "AVG" not in x ]
        self.fnames = sorted(
            [path.split(ffile)[1] for ffile in self.ffiles]
        )
        if not self.importer:
            return
        self.importer.w_files.options = self.fnames

def setup_inspector(folder, observer, importer):
    '''Sets up an Inspector with a file handler'''
    if not path.isdir(folder):
        if debug: print('setup_inspector called with invaild fpath')
        return observer, importer, None
    
    event_handler = FileHandler(folder)

    # On the first run the importer is None. From Then on we update it.

    if not importer:
        if debug: print('creating new importer')
        try:
            importer = sfg2d.widgets.Importer(folder)
            importer()
        except OSError:
            if debug: print('OSError during creation of importer')
        except IndexError:
            if debug: print('IndexError during creation of importer')
        
    # Start file observer to monitor filechanges in new folder
    observer.schedule(event_handler, folder, recursive=False)
    observer.start()
    return observer, importer, event_handler

def update_inspector(new):
    '''Callback for the W_FOLDER observer'''
    #return setup_inspector(W_FOLDER.value, OBSERVER, IMPORTER)
    global OBSERVER, IMPORTER, EVENT_HANDLER
    
    if not path.isdir(W_FOLDER.value):
        W_FOLDER.layout.border = '3px red dotted'
        if debug: print('update_ispector with invalid path. Skipping')
        return
    
    W_FOLDER.layout.border = ''
    
    if OBSERVER.isAlive():
        if debug: print('Killing old OBSERVER')
        OBSERVER.unschedule_all()
        OBSERVER.stop()
        OBSERVER.join()
    OBSERVER = Observer()

    EVENT_HANDLER = FileHandler(W_FOLDER.value)
    if not isinstance(IMPORTER, sfg2d.widgets.Importer):
        if debug: print('creating new IMPORTER')
        try:
            IMPORTER = sfg2d.widgets.Importer(W_FOLDER.value)
            IMPORTER()
        except OSError:
            return
        
    EVENT_HANDLER.importer = IMPORTER
    IMPORTER.ffolder = W_FOLDER.value
    IMPORTER.w_files.options = EVENT_HANDLER.fnames
    

W_FOLDER = Text(
    description = 'Folder:',
    value = 'D:/das/2016/09/30',
    )

display(W_FOLDER)
if not path.isdir(W_FOLDER.value):
    W_FOLDER.layout.border = '3px red dotted'
OBSERVER, IMPORTER, EVENT_HANDLER = setup_inspector(W_FOLDER.value, Observer(), None)
W_FOLDER.observe(update_inspector, 'value')
