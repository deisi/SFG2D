from numpy import log10, floor, abs, round
from .detect_peaks import detect_peaks
from .metadata import get_metadata_from_filename
from .static import (
    wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers,
    ir_wavenumbers_to_nm, savefig, get_interval_index, find_nearest_index
)
from .consts import (
    X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX,
    PP_INDEX, PIXEL, FRAME_AXIS_INDEX
)
from .filter import double_resample, replace_pixel


def round_sig(x, sig=2):
    """Round to sig number of significance."""
    return round(x, sig-int(floor(log10(abs(x))))-1)


def round_by_error(value, error, min_sig=2):
    """Use value error pair, to round.

    value: Value of the variable
    error: uncertaincy of the variable

    """

    sig_digits = int(floor(log10(abs(value)))) - int(floor(log10(abs(error)))) + 1
    # Kepp at least minimal number of significant digits.
    if sig_digits <= min_sig: sig_digits = min_sig
    return round_sig(value, sig_digits), round_sig(error, 1)
