"""Module to import data from the victor controller."""
import os, datetime

import numpy as np

from ..utils.consts import PIXEL # x-pixel of the camera
from ..utils.metadata import MetaData
SPECS = 3 # Number of binned spectra.

def get_from_victor_controller(fpath):
    raw_data = np.genfromtxt(fpath, dtype='long')
    pp_delays = np.array([0])

    num_rows, num_columns = raw_data.shape

    # File is just a simple scan.
    if num_rows == PIXEL and num_columns == 4:
        return np.array([[raw_data.T[1:]]]), pp_delays

    # Check that we can read the data shape
    if (num_columns)%(SPECS+1) != 0:
        raise IOError("Cant read data in %s" % fpath)

    num_pp_delays = num_rows%PIXEL
    # The first colum is only pixel number
    num_repetitions = num_columns//(SPECS+1)

    # Get pp_delay values
    pp_delay_row_indexes = [i * PIXEL + i for i in range(num_pp_delays)]
    # Get rows that count the pixels
    pixel_column_indeces = [i*(SPECS+1) for i in range(num_repetitions)]
    pp_delays = raw_data[pp_delay_row_indexes, 0]
    # pp_delays rows are not needed any more
    raw_data = np.delete(raw_data, pp_delay_row_indexes, 0)
    # pixel columns are not needed any more
    raw_data = np.delete(raw_data, pixel_column_indeces, 1)

    # Init container for the result.
    ret = np.zeros((num_pp_delays, num_repetitions, SPECS, PIXEL))

    for rep_index in range(num_repetitions):
        for pp_delay_index in range(num_pp_delays):
            column_slice = slice(PIXEL*pp_delay_index, PIXEL*pp_delay_index + PIXEL)
            row_slice = slice(rep_index*SPECS, rep_index*SPECS+SPECS)
            ret[pp_delay_index, rep_index] = raw_data[column_slice, row_slice].T

    return ret, pp_delays

def read_header(fpath):
    """Read informaion from fileheader adn return as dictionary."""
    ret = {}
    with open(fpath) as f:
        for line in f:
            if line[0] is not "#":
                break
            # Strip comment marker
            line = line[2:]
            name, value = line.split("=")
            # Strip newline
            ret[name] = value[:-1]
    return ret

def translate_header_to_metadata(header_dict):
    """Translate the header dictionary to metadata dictionary."""
    ret = {}
    for key in header_dict:
        value = header_dict[key]
        if 'ExposureTime' in key:
            _, unit = key.split(" ")
            if "[s]" in unit:
                unit = "seconds"
            print(unit, value)
            ret["exposure_time"] = datetime.timedelta(**{unit : float(value)})

        if 'Syringe Pos' in key:
            ret["syringe_pos"] = int(value)

        if "Timefile" in key:
            ret["timefile"] = value

        # Wavelength => CentralWavelength
        if "Wavelength" in key:
            ret["central_wl"] = int(value)


    return ret
