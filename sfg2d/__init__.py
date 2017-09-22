__version__ = '0.6.0'

from . import io, utils, plotting, widgets, fit

from .core import (SfgRecord, concatenate_list_of_SfgRecords,
                   SfgRecords_from_file_list)

from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save

from .utils import detect_peaks, consts, get_metadata_from_filename, \
    wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, \
    ir_wavenumbers_to_nm, X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX, PP_INDEX,\
    PIXEL, FRAME_AXIS_INDEX, savefig, get_interval_index, double_resample

from .utils.metadata import time_scan_time

from .utils.static import conv_gaus_exp_f as double_decay

from .utils.filter import double_resample, replace_pixel

from .plotting import (multipage_pdf, plot_model_trace, plot_record_contour,
                       plot_record_static, plot_trace_fit, plot_contour,
                       plot_trace, plot_spec)

from .models import FourLevelMolKin
