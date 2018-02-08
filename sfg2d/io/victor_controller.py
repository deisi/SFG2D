"""Module to import data from the victor controller."""
import datetime
import numpy as np

from sfg2d.utils import PIXEL, SPECS
from sfg2d.utils.metadata import get_unit_from_string


def get_from_victor_controller(fpath, **kwargs):
    """Import data from victor controller

    kwargs are passed to np.genfromtxt"""
    try:
        raw_data = np.genfromtxt(fpath, dtype='long', **kwargs)[:, 1:]
    except ValueError:
        raise IOError("Scan was interrupted. Plz give usecols kwarg to genfromtxt.")

    with open(fpath) as file:
        for line in file:
            if '# Timedelay=' in line:
                pp_delays = np.array(line[12:-1].split('\t'), dtype=int)
                break
            if line[0] is not '#':
                break

    num_rows, num_columns = raw_data.shape

    # File is just a simple scan.
    if num_rows == PIXEL and num_columns == 3:
        return np.array([[raw_data.T]]), pp_delays

    # Check that we can read the data shape
    if (num_columns)%SPECS != 0:
        raise IOError("Cant read data in %s" % fpath)

    num_pp_delays = num_rows//PIXEL

    # The first colum is only pixel number
    num_repetitions = num_columns//SPECS

    # Init container for the result.
    ret = np.zeros((num_pp_delays, num_repetitions, SPECS, PIXEL))

    for rep_index in range(num_repetitions):
        for pp_delay_index in range(num_pp_delays):
            column_slice = slice(PIXEL*pp_delay_index, PIXEL*pp_delay_index + PIXEL)
            row_slice = slice(rep_index*SPECS, rep_index*SPECS+SPECS)
            ret[pp_delay_index, rep_index] = raw_data[column_slice, row_slice].T

    return ret, pp_delays


def read_header(fpath):
    """Read informaion from fileheader and return as dictionary."""
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
        if 'Gain' == key:
            ret['gain'] = value

        elif "Output Amplifier" == key:
            ret['output amplifier'] = value

        elif "HS_speed" in key:
            unit = get_unit_from_string(key)
            ret['hs_speed'] = (value, unit)

        elif 'ExposureTime' in key:
            _, unit = key.split(" ")
            if "[s]" in unit:
                unit = "seconds"
            ret["exposure_time"] = datetime.timedelta(**{unit : float(value)})

        elif 'HBin' == key:
            ret['hbin'] = {'ON': True}.get(value, False)

        # Wavelength => CentralWaveleng
        elif "Central-Wavelength" == key:
            ret["central_wl"] = int(value)

        elif "vis-Wavelength" == key:
            ret["vis_wl"] = int(value)

        elif 'Syringe Pos' in key:
            ret["syringe_pos"] = int(value)

        elif "Timefile" in key:
            ret["timefile"] = value

        elif "Cursor" == key:
            ret['cursor'] = tuple([int(elm) for elm in value.split('\t')])

        elif 'x-mirror' == key:
            ret['x-mirror'] = {'ON': True}.get(value, False)

        elif 'calib Coeff' == key:
            ret['calib Coeff'] = tuple([float(elm) for elm in value.split('\t')])

    return ret
