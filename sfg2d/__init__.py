__version__ = '0.6.0'

from . import io, utils, plotting, widgets, fit

from .core import SfgRecord, concatenate_list_of_SfgRecords

from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save

from .utils import detect_peaks, consts, get_metadata_from_filename, \
    wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, \
    ir_wavenumbers_to_nm, X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX, PP_INDEX,\
    PIXEL, FRAME_AXIS_INDEX, savefig, get_interval_index, double_resample

from .utils.metadata import MetaData, time_scan_time

from .plotting import plot, plot_time, multipage_pdf, contour
