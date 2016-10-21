__version__ = '0.5.0'

from . import io, utils, plotting, widgets

from .core.scan import Scan, TimeScan

from .io import AllYouCanEat
from .io.spe import PrincetonSPEFile3
from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save
from .io.ntb import NtbFile

from .utils import detect_peaks
from .utils.metadata import MetaData, time_scan_time, get_metadata_from_filename
from .utils.static import wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, ir_wavenumbers_to_nm, savefig

from .plotting import plot
