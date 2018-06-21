import warnings
import os
from numpy import log10, floor, abs, round, poly1d, load
from .detect_peaks import detect_peaks
from .metadata import get_metadata_from_filename
from .static import (
    wavenumbers_to_nm, nm_to_wavenumbers, nm_to_ir_wavenumbers,
    ir_wavenumbers_to_nm, savefig, get_interval_index, find_nearest_index
)
from sfg2d.utils.config import CONFIG

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


def pixel_to_nm(
        x,
        central_wl,
):
    """ transform pixel to nanometer

    Parameters
    ----------
    central_wl : int
        central wavelength of the camera in nm
    params_file_path: Optinal file path to calibration parameter file
        If None given, a default is loaded.
    """

    pixel_to_nm = poly1d(CONFIG['CALIB_PARAMS']) + central_wl - CONFIG['CALIB_CW']
    return pixel_to_nm(x)

def nm_to_pixel(x, central_wl):
    """ transform nm to pixel coordinates for central wavelength

    Parameters
    ----------
    x : array like
        nm to transform in pixel
    central_wl : int
        central wavelength of the camera

    Returns
    -------
    num or array of x in pixel coordinates
    """

    params_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
    )
    params = load(params_file_path)
    calib_cw = int(params_file_path[-7:-4])
    if len(params) > 2:
        params = params[-2:]
    if len(params) < 2:
        warnings.Warn("Can't use constant calibration")
    return x - params[1] - central_wl + calib_cw/params[0]

def get_dataframe(record, seconds, kwargs_track=None):
    """Creates a dataframe from track and time data.
    seconds: number of paused seconds between frames. This is not saved by spe file

    the difference of frame time is read from .spe metadata information.
    """
    import pandas as pd
    from datetime import timedelta

    if not kwargs_track:
        kwargs_track = {}
    ydata = record.track(**kwargs_track).squeeze()
    start_time = record.metadata['date']
    measurment_time = [start_time + timedelta(seconds=seconds)*i for i in range(len(ydata))]
    df = pd.DataFrame(ydata, index=measurment_time)
    return df
