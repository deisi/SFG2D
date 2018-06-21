"""IO Module to import data from the veronica labview programm """

import numpy as np
from sfg2d.utils.config import CONFIG
SPECS = CONFIG['SPECS']
PIXEL = CONFIG['PIXEL']

debug=0


def get_from_veronika(fpath):
    """Read files saved by veronika labview programm

    The function reads a file from veronika labview,
    and returns it as a unified 4 dimensional numpy array. And an
    array with all the pump probe time delays.

    Parameters
    ----------
    fpath: str
        Path to load data from.

    Returns
    -------
    tuple of two arrays.
    First array is:
        4 Dimensional numpy array with:
            0 index pp_delays,
            1 index number of repetitions
            2 index number of y-pixel/spectra/bins
            3 index x-pixel number
    """
    col_mult = SPECS + 3  # Multiplicity of colums aka blocksize
    raw_data = np.genfromtxt(fpath)
    pp_delays = np.array([0])

    # File is just a simple scan
    if raw_data.shape == (PIXEL, col_mult):
        ret = np.array([[raw_data.T[1:4]]])
        return ret, pp_delays

    # Check that the shape is readable
    if (raw_data.shape[1]) % col_mult != 0:
        raise IOError("Cant read data in %s" % fpath)

    # We delete the first scan block, because it
    # is the average of the others.
    raw_data = raw_data[:, col_mult:]

    # Delete the all 0 lines, that mark the end of a block
    if debug > 2:
        print("Verify, that only empty values have beed removed:")
        print(raw_data[PIXEL+1::PIXEL+2])
    raw_data = np.delete(raw_data, slice(PIXEL+1, None,PIXEL+2), 0)

    num_rows, num_columns = raw_data.shape

    # Every 1600 lines there is an additional line with the pp_delays
    num_pp_delays = num_rows % (PIXEL)
    # The first colum is only pixel number
    num_repetitions = num_columns//col_mult
    pp_delays = raw_data[::PIXEL+1][:, 0]
    # Delete pp_delay rows
    raw_data = np.delete(raw_data, slice(None, None, PIXEL+1), 0)

    # Delete pixel number columns
    raw_data = np.delete(raw_data, slice(None, None, col_mult), 1)

    # Delete lines, that are the division or avg of other lines
    raw_data = np.delete(raw_data, slice(None, None, col_mult-1), 1)
    raw_data = np.delete(raw_data, slice(SPECS, None, col_mult-2), 1)

    # Init container for the result.
    ret = np.zeros((num_pp_delays, num_repetitions, SPECS, PIXEL), dtype=raw_data.dtype)
    for rep_index in range(num_repetitions):
        for pp_delay_index in range(num_pp_delays):
            column_slice = slice(PIXEL*pp_delay_index, PIXEL*pp_delay_index + PIXEL)
            row_slice = slice(rep_index*SPECS, rep_index*SPECS+SPECS)
            ret[pp_delay_index, rep_index] = raw_data[column_slice, row_slice].T

    ret = ret.astype('long')
    return ret, pp_delays
