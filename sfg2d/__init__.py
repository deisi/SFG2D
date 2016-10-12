__version__ = '0.4.0'

from . import io, utils, plotting, widgets
from .core.scan import Scan, TimeScan
from .utils.metadata import MetaData

from .io.spe import PrincetonSPEFile3
from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save
from .utils.static import wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, ir_wavenumbers_to_nm, savefig
