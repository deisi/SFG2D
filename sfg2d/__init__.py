__version__ = '0.5.0'

from . import io, utils, plotting, widgets, fit

from .core.scan import Scan, TimeScan

#import .fit

#from .io import AllYouCanEat, PrincetonSPEFile3, NtbFile
from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save
from .io.allYouCanEat import AllYouCanEat

from .utils import detect_peaks
from .utils.metadata import MetaData, time_scan_time, get_metadata_from_filename
from .utils.static import wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, ir_wavenumbers_to_nm, savefig

from .plotting import plot, plot_time
