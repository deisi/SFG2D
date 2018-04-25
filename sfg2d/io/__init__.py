#from . import veronica, victor_controller
from .spe import PrincetonSPEFile3
from .ntb import NtbFile
from .veronica import get_from_veronika
from .victor_controller import (
    get_from_victor_controller,
    read_header,
    translate_header_to_metadata
)
from ..utils.metadata import get_metadata_from_filename

from os import path
import warnings
import numpy as np


def import_data(fname, type=None):
    """Import the data."""
    # Pick import function according to data type automatically.
    metadata = {}
    if isinstance(fname, str):
        one_file = fname
    else:
        one_file = fname[0]

    if not type:
        type = type_data(one_file)

    try:
        metadata = get_metadata_from_filename(one_file)
    # TODO Refactor this, if I would program better this
    # would not happen
    except ValueError:
        msg ='ValueError while trying to extract metadata from filepath.'\
            '/nSkipping'
        warnings.warn(msg)

    if type == "spe":
        if isinstance(fname, str):
            sps = [PrincetonSPEFile3(fname)]
        else:
            sps = [PrincetonSPEFile3(name) for name in fname]
        sp = sps[0]
        metadata['central_wl'] = sp.central_wl
        metadata['exposure_time'] = sp.exposureTime
        metadata['gain'] = sp.gain
        metadata['sp_type'] = 'spe'
        metadata['date'] = sp.date
        metadata['tempSet'] = sp.tempSet
        metadata['wavelength'] = sp.wavelength
        metadata['calib_poly'] = sp.calib_poly
        #metadata['NumFrames'] = sp.NumFrames
        #metadata['ydim'] = sp.ydim
        #metadata['xdim'] = sp.xdim
        return {'data': sps, 'metadata':  metadata, 'type': type}

    if type == "npz":
        if isinstance(fname, str):
            return {'type': type, 'data': np.load(fname), 'metadata':  metadata,}
        raise NotImplementedError('List implementation for npz files not implemented')

    if type == "veronica":
        if isinstance(fname, str):
            return {'data': get_from_veronika(fname), 'type': type, 'metadata':  metadata,}
        raise NotImplementedError('List implement for veronica files not implemented')

    if type == "victor":
        # Read metadata from file header.
        if isinstance(fname, str):
            header = read_header(fname)
            metadata = {**metadata, **translate_header_to_metadata(header)}
            return {'data': get_from_victor_controller(fname), 'type': type, 'metadata':  metadata,}
        # A list of files was passed
        else:
            header = read_header(fname[0])
            metadata = {**metadata, **translate_header_to_metadata(header)}
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
    if start_of_data.shape[1] == 4:
        return 'victor'

    elif start_of_data.shape[1] == 6:
        return 'veronica'
    else:
        # Check if we have a header.
        # Only data from victor_controller has # started header.
        with open(fname) as f:
            line = f.readline()
            if line[0] == "#":
                # First line is pixel then 3 spectra repeating
                if (start_of_data.shape[1] - 1) % 3 == 0:
                    return 'victor'
            else:
                if start_of_data.shape[1] % 6 == 0:
                    return 'veronica'
    raise IOError("Cant understand data in %f" % fname)



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
