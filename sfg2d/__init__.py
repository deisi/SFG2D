__version__ = '0.6.0'

# CONFIG Get loaded first because its often used by other modules.
from .utils.config import CONFIG

from . import io, utils, plot, widgets, models

from .core import (SfgRecord, concatenate_list_of_SfgRecords,
                   SfgRecords_from_file_list, import_sfgrecord)

from .utils import detect_peaks, get_metadata_from_filename, \
    wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers, \
    ir_wavenumbers_to_nm, savefig, get_interval_index, \
    double_resample

from .utils.metadata import time_scan_time

from .utils.static import conv_gaus_exp_f as double_decay

from .utils.filter import double_resample, replace_pixel

from .plotting import (multipage_pdf, plot_model_trace, plot_record_contour,
                       plot_record_static, plot_trace_fit, plot_contour,
                       plot_trace, plot_spec, figures2pdf, savefig_multipage)

from .models import FourLevelMolKinM as FourLevelMolKin, fit_model
