__version__ = '0.3.1'

from . import io, utils, plotting, widgets
from .core.scan import Scan, TimeScan
from .utils.metadata import MetaData

from .io.spe import PrincetonSPEFile3
from .io.veronica import read_auto, read_time_scan, read_scan_stack, read_save


def savefig(filename, **kwargs):
    import matplotlib.pyplot as plt
    '''save figure as pgf, pdf and png'''
    plt.savefig('{}.pgf'.format(filename), **kwargs)
    plt.savefig('{}.pdf'.format(filename), **kwargs)
    plt.savefig('{}.png'.format(filename), **kwargs)
