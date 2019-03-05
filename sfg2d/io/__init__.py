#from . import veronica, victor_controller
import logging
from .spe import PrincetonSPEFile3
from .ntb import NtbFile
from .veronica import get_from_veronika
from .victor_controller import (
    get_from_victor_controller,
    read_header,
)
from ..utils.metadata import get_metadata_from_filename
from sfg2d.utils.config import CONFIG

from os import path
import warnings
import numpy as np

SPECS = CONFIG['SPECS']
logger = logging.getLogger(__name__)

def import_data(fname, type=None):
    """Import the data."""
    # Pick import function according to data type.
    metadata = {}
    if isinstance(fname, str):
        one_file = fname
    else:
        one_file = fname[0]

    if not type:
        type = type_data(one_file)

    # TODO Refactor this so the try is not needed
    try:
        metadata = get_metadata_from_filename(one_file)
    except ValueError:
        msg ='ValueError while trying to extract metadata from filepath.'\
            '/nSkipping'
        warnings.warn(msg)

    if type == "spe":
        if isinstance(fname, str):
            sps = [PrincetonSPEFile3(fname)]
        else:
            sps = [PrincetonSPEFile3(name) for name in fname]
        # Metadata is read of the first file, because ther is no clear mapping
        sp = sps[0]
        metadata = {**metadata, **sp.metadata}
        metadata['sp_type'] = 'spe'
        return {'data': sps, 'metadata':  metadata, 'type': type}

    if type == "npz":
        if isinstance(fname, str):
            return {'type': type, 'data': np.load(fname), 'metadata':  metadata,}
        if isinstance(fname, list):
            msg = 'Only concatenates rawData, base and norm.'
            msg += 'The rest is taken from first entry.'
            logger.warn(msg)
            ret = {'type': type, 'metadata': metadata}
            datas = []
            for elm in fname:
                datas.append(np.load(elm))

            # npz imports are almost dicts. However, we cant overwrite
            # the npz import because this could lead to data beeing written
            # on the hdd without us wanting it. This is an import function
            # not a saving function. Thus the ret dict, that encapsulates the
            ret['data'] = {}
            for key in datas[0].keys():
                ret['data'][key] = datas[0][key]

            for key in ('rawData', 'norm', 'base'):
                ret['data'][key] = np.concatenate([datas[i][key] for i in range(len(datas))], 1)

            for elm in datas:
                elm.close()
            return ret
        raise NotImplementedError('.npz files must be passed individually or as list.')

    if type == "veronica":
        if isinstance(fname, str):
            return {'data': get_from_veronika(fname), 'type': type, 'metadata':  metadata,}
        raise NotImplementedError('List implement for veronica files not implemented')

    if type == "victor":
        # Read metadata from file header.
        if isinstance(fname, str):
            metadata = {**metadata, **read_header(fname)}
            return {'data': get_from_victor_controller(fname), 'type': type, 'metadata':  metadata,}
        # A list of files was passed
        else:
            metadata = {**metadata, **read_header(fname[0])}
            raw_data, pp_delays = get_from_victor_controller(fname[0])
            for name in fname[1:]:
                r, p = get_from_victor_controller(name)
                raw_data = np.append(raw_data, r, axis=1)
            return {'data': (raw_data, pp_delays), 'type': type, 'metadata':  metadata,}

    msg = "I cannot understand the datatype of {}".format(fname)
    raise NotImplementedError(msg)


def type_data(fname):
    """
    Get the data type of fname.

    **Arguments:**
      - **fname**: path to a data file

    **Returns:**
    string to identify the type of the data.
    """
    fhead, ftail = path.split(fname)

    # spe is binary and we hope its not named wrongly
    if path.splitext(ftail)[1] == '.spe':
        return 'spe'

    # Compressed binary version.
    if path.splitext(ftail)[1] == '.npz':
        return 'npz'

    # We open the file and by looking at the
    # first view lines, we can see if that is a readable file
    # and what function is needed to read it.
    start_of_data = np.genfromtxt(fname, max_rows=3, dtype="long")
    with open(fname) as f:
        line = f.readline()
        if line[0] == "#":
            # First line is pixel then 3 spectra repeating
            if (start_of_data.shape[1] - 1) % 3 == 0:
                return 'victor'
        else:
            return 'veronica'


def load_fitarg(fp):
    """load the fit results from fp"""
    from json import load

    with open(fp) as json_data:
        ret = load(json_data)
    return ret


def load_fitarg_and_minuit(fp, fit_func, migrad=True):
    """Load fit results from fp and add to scan.

    Parameters
    ----------
    fp : string
        filepath
    fit_func : The function of the fitarg
    migrad : Boolean
        If true the fit will be performed again to offer additional properties

    Returns
    -------
    fitarg : dictionary
    minuit : Minuit obj with the fit
    """
    from iminuit import Minuit

    fitarg = load_fitarg(fp)
    minuit = Minuit(fit_func, **fitarg, pedantic=False)
    if not migrad:
        return fitarg, minuit
    print('**********************************************************************')
    print('Fitting with values from %s' % fp)
    minuit.migrad()
    return fitarg, minuit


def load_fitarg_minuit_chi2(fp, fit_func, x, y, migrad=True, **kwargs):
    """Load fitresult and construct minuit and chi2 from it.

    Parameters
    ----------
    fp : str
        path to fitarg file
    fit_func : function
        The function the fit is performed with

    x : array
        x-data
    y : array
        y-data

    kwargs
    ------
    migrad : Boolean
        The fit is directly performed
    Other kwargs are passed to the Chi2Regression.
    Most important one is
    **y_err : The y_err of the fit
    For the rest see probfit.Chi2Regression

    Returns
    -------
    fitarg: dictionary
        Fit results
    minuit : Minuit ob of the fit
    chi2 : Probfit.Chi2Regression
        Probfit obj. to determine the quality of the fit.
        This makes only sense, when at least an y_err was provided. Otherwise this is
        a useless thing
    """
    from probfit import Chi2Regression

    fitarg, minuit = load_fitarg_and_minuit(fp, fit_func, migrad)
    chi2 = Chi2Regression(
        fit_func,
        x,
        y,
        **kwargs
    )
    return fitarg, minuit, chi2


def ac_data_hj(fname, **kwargs):
    """Function to read humidity and temperature data from hansjoerg"""
    from pandas import read_csv, to_datetime

    return read_csv(
        fname, sep=';',
        index_col=0,
        decimal=',',
        parse_dates=True,
        date_parser=lambda dt: to_datetime(dt, format='%d.%m.%Y %H:%M:%S'),
        **kwargs
       )


def ac_data_vic(fname, **kwargs):
    """Read data from viktor temperature and humidity controller."""
    import pandas as pd

    return pd.read_csv(
        fname,
        index_col=1,
        parse_dates=True,
        names=['Temperature\Humidity Graph', 'Time', 'Temp', 'High Alarm',
               'Low Alarm', 'Hum', 'High Alarm rh', 'Low Alarm rh',
               'dew point', 'High Alarm dew', 'Low Alarm dew'],
        skiprows=12,
        )
