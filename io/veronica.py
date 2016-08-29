"""IO Module to handle data from the veronica labview programm """

import os 
import re
import copy
#import logging
import warnings

from pandas import read_csv, MultiIndex, Series
import numpy as np


from ..core.scan import Scan, TimeScan
from ..utils.static import nm_to_ir_wavenumbers
from ..utils.metadata import get_metadata_from_filename


names = (
    'pixel',
    'spec_0',
    'spec_1',
    'spec_2',
    'ratio_0',
    'ratio_1'
)
PIXEL = 1600

def pixel_to_nm(x, central_wl):
    """ transform pixel to nanomenter

    Parameters
    ----------
    central_wl : int
        central wavelength of the camera in nm"""
    
    params_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
    )
    params = np.load(params_file_path)
    calib_cw = int(params_file_path[-7:-4])
    pixel_to_nm = np.poly1d(params) + central_wl - calib_cw
    return pixel_to_nm(x)

def read_save(fpath, **kwargs):
    """read files saved with veronica save button """

    ret = read_csv(
        fpath,
        sep = '\t',
        header = None,
        names = names,
        #index_col = 'pixel',
        usecols = [0, 1, 2, 3],
        #dtype = np.int16,
        **kwargs
    )

    # metadata based on filename makes the calibration
    metadata = get_metadata_from_filename(fpath)
    pixel = np.arange(PIXEL)
    nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
    wavenumber = np.round(nm_to_ir_wavenumbers(
        nm, up_wl=metadata['vis_wl']
    ), decimals = 1)

    #index = MultiIndex.from_arrays(
    #    (pixel, nm, wavenumber), names=('pixel', 'nm', 'wavenumber')
    #)

    #ret.index = index
    
    ret = ret.astype('int16')

    ret.set_index(wavenumber, inplace=True)
    ret.index.name = 'wavenumber'
    ret.index = ret.index.sort_values()
    ret.sort_index(inplace=True)

    ret = Scan(ret, metadata=copy.deepcopy(metadata))

    return ret

def read_scan_stack(fpath, **kwargs):
    """import veronica scan files containing only repetition """
    
    ret = read_csv(
        fpath,
        sep = '\t',
        header = None,
        skiprows = 1,
        skipfooter = 1,
        engine = 'python',
        **kwargs
    )

    # Check for correct data shape
    if ret.shape[1]%6 is not 0:  #and ret.shape[0] is not 1602:
        raise IOError("Cant understand shape of data in %s"%fpath)

    # number of spectrum repetition
    reps = ret.shape[1]//6

    def _iterator(elm):
        return np.array(
            # ditch first repetition because it is the AVG
            [elm(i) for i in range(1, reps)]
        ).flatten().tolist()
        
    # Keep only interesting columns
    colum_inds = [0] +  _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
    ret = ret[ret.columns[colum_inds]]

    # Set colum names
    names = ['pixel'] + _iterator(lambda i: ['spec_0', 'spec_1', 'spec_2'])
    ret.columns = names

    # metadata based on filename makes the calibration
    metadata = get_metadata_from_filename(fpath)
    pixel = np.arange(1600)
    nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
    wavenumber = np.round(nm_to_ir_wavenumbers(
        nm, up_wl=metadata['vis_wl']
    ), decimals = 1)

    #index = MultiIndex.from_arrays(
    #    (pixel, nm, wavenumber), names=('pixel', 'nm', 'wavenumber')
    #)

    #ret.index = index

    # Data types are integers
    ret = ret.astype('int16')    

    # Use pixel as index
    #ret.set_index("pixel", inplace=True)
    #ret['wavenumber'] = Series(wavenumber, index=ret.index)
    ret.set_index(wavenumber, inplace=True)
    ret.index.name = 'wavenumber'
    ret.sort_index(inplace=True)
    #ret['wavenumber'] = wavenumber
    
    # Link Repeated Columns together
    return Scan(ret, metadata=copy.deepcopy(metadata))

def read_time_scan(fpath, **kwargs):
    """ """
    ret = read_csv(
        fpath,
        sep = '\t',
        header = None,
        **kwargs
    )

    # Check for correct data shape
    if ret.shape[1] % 6 is not 0  and ret.shape[1] % 1602 is not 0:
        raise IOError("Cant understand shape of data in %s"%fpath)

    # number of spectrum repetition
    reps = ret.shape[1]//6
    # number of times
    times = ret.shape[0]//1602

    def _iterator(elm):
        return np.array(
            # ditch first iteration because it the AVG
            [elm(i) for i in range(1, reps)]
        ).flatten().tolist()
        
    # Remove uninteresting columns
    colum_inds = [0] + _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
    ret = ret[ret.columns[colum_inds]]

    # Data type is integer
    ret = ret.astype('int16')
    
    # Set colum names
    names = ['pixel'] +_iterator(lambda i: ['spec_0', 'spec_1', 'spec_2'])
    ret.columns = names

    # time delays so we can use them for multiaxes
    pp_delays = ret.iloc[0::1602, 0]

    # drop lines with pp_delays
    ret.drop(pp_delays.index, inplace=True)

    # remove empty lines like the 1600 row
    ret = ret[~np.all(ret == ret.iloc[1600], 1)]

    # drop pixel column. We add it as one axes of the multiaxes later
    #ret.drop('pixel', axis=1, inplace=True)
    
    # metadata based on filename makes the calibration and are used to
    # index the dataframe
    metadata = get_metadata_from_filename(fpath)
    pixel = np.arange(PIXEL)
    nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
    wavenumber = np.round(nm_to_ir_wavenumbers(
        nm, up_wl=metadata['vis_wl']
    ), decimals = 1)

    #ret['wavenumber'] = np.repeat([wavenumber], len(pp_delays), axis=0) .flatten()

    # make indeces
    pp_delays = pp_delays.as_matrix()
    index = MultiIndex.from_product(
        (pp_delays, wavenumber),
        names=('pp_delay', 'wavenumber')
    )
    ret.index = index
    ret.sort_index(inplace=True)

    #ret = ret.set_index([nm, wavenumber], append=True)
    #ret.index.names = ['pp_delays', 'pixel', 'nm', 'wavenumber']

    # Link Repeated Columns together
    return TimeScan(ret, metadata=copy.deepcopy(metadata))

def read_auto(fpath, **kwargs):
    """ use fpath and datashape to automatically determine data type
    of fpath and use according read function to import data"""
    folder, ffile = os.path.split(fpath)
    metadata = get_metadata_from_filename(fpath)

    # check if name determines spectrum type
    if metadata['sp_type'] is 'sp' or \
       metadata['sp_type'] is 'sc' or \
       metadata['sp_type'] == 'ts':
        pass

    else:
        warnings.warn('cant determine spectrum type of data by filename.'
                      'Trying to determine datatype from content.'
                      'This is much slower')
        ret = read_csv(
            fpath,
            sep = '\t',
            header = None,
        )
        if ret.shape is (1600, 6):
            metadata['sp_type'] = 'sp'
        elif ret.shape[0] == 1602 and ret.shape[1]%6 is 0:
            metadata['sp_type'] = 'sc'
        elif ret.shape[0]%1602 is 0 and ret.shape[1]%6 is 0:
            metadata['sp_type'] = 'ts'

    # simple spectrum
    if metadata['sp_type'] == 'sp':
        ret = read_save(fpath, **kwargs)

    # scan
    elif metadata['sp_type'] == 'sc':
        ret = read_scan_stack(fpath, **kwargs)

    # time scan
    elif metadata['sp_type'] == 'ts':
        ret = read_time_scan(fpath, **kwargs)
        
    ret.df.drop("pixel", inplace=True, axis=1)
    # Needed in the case of content import. Else no harm
    ret.metadata['sp_type'] = copy.deepcopy(metadata['sp_type'])
    return ret

        
def get_scan(fpath, fbase=None, fnorm=None, sub_base=True, div_norm=True):
    ret = read_auto(fpath)

    if fbase:
        ret.base = read_auto(fbase).med
        if sub_base:
            ret.sub_base(inplace=True)

    if fnorm:
        ret.norm = read_auto(fnorm).med
        if div_norm:
            ret.normalize(inplace=True)

    return ret

def get_time_scan(fpath, base = None, norm = None):
    ret = read_time_scan(fpath)
    ret.df.drop("pixel", axis=1, inplace=True)
    if getattr(base):
        ret.base = base
        ret.sub_base(inplace=True)

    if getattr(norm):
        ret.norm = norm
        ret.normalize(inplace=True)
