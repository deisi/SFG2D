import numpy as np
import warnings
import pandas as pd
import os

from ..utils.read_spe import read_spe
from ..core.scan import Scan
from ..utils.metadata import MetaData

PIXEL = 1600


def read_spe_as_scan(ffile):
    if not os.path.isfile(ffile):
        raise IOError("%s does not exist" % ffile)
    
    sp = read_spe(ffile)
    metadata = MetaData
    metadata["uri"] = os.path.abspath(ffile)

    y_pixels = len(sp['data'][0])
    if y_pixels != 1:
        raise IOError("%s is not a binned spectrum" % ffile)
    
    data = np.reshape(np.array(sp['data'], dtype=np.int),
                   (len(sp['data']), PIXEL))
    index = np.arange(PIXEL)
    columns = ['spec_0' for i in range(data.shape[0])]

    
    params = np.array(sp['XCALIB']['polynom_coeff'])
    # check if coefficients are readable
    if all(params == 0):
        warnings.warn("No calibration parameters in %s" % ffile)
    else:
        # Parameters needed to be in decreasing oder fo numpy ...
        params = params[where(params != 0)][::-1]
        pixel_to_nm = np.poly1d(params)

        
    ret = Scan(pd.DataFrame(data=data.T, index=index, columns=columns))
    return ret

def read_spe_img(ffile):
    if not os.path.isfile(ffile):
        raise IOError("%s does not exist" % ffile)
    
    sp = read_spe(ffile)
    metadata = MetaData
    metadata["uri"] = os.path.abspath(ffile)
    # check shape
    if len(sp['data']) != 1:
        raise NotImplementedError
    
    y_pixels = len(sp['data'][0])
    # Check if full chip
    if (y_pixels != 200 and y_pixels != 400):
        raise IOError("%s is not a full camera img\nGot %i y_pixels" % (ffile , y_pixels))
    
    data = np.reshape(np.array(sp["data"], dtype=np.int),
                          (y_pixels, PIXEL))
    index = np.arange(PIXEL)

    ret = pd.DataFrame(data.T, index)
    return ret
    
