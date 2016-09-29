"""Module to transform spe files into the SFG2D Scan data structure"""

from numpy import arange
from pandas import DataFrame, Index
from os import path

from ..utils.spe import PrincetonSPEFile3
from ..core.scan import Scan
from ..utils.metadata import MetaData

def read_spe_as_scan(fname):
    """Read spe file as scan. If data wasn't binned, this will bin ydim.

    Parameters
    ----------
    fname : str
        Path to the file to import
    """
    if not path.isfile(fname):
        raise IOError("%s does not exist" % fname)
    
    sp = PrincetonSPEFile3(fname)
    metadata = MetaData
    metadata["uri"] = path.abspath(fname)
    metadata["central_wl"] = sp.central_wl
    data = sp.data.mean(1)
    # Each frame is just the repetition of the same measurment
    # In the SFG2D module this translates into columns wit the same
    # name
    columns = ['spec_0' for i in range(sp.NumFrames)]

    # If possible add wavelength as index
    index = None
    if all(sp.wavelength != np.arange(sp.xdim)):
        index = Index(sp.wavelength, name="wavelength")
     
    ret = Scan(DataFrame(data=data.T, index=index, columns=columns), metadata=metadata)
    
    return ret    
