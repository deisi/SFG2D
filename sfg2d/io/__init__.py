from . import veronica
from .spe import PrincetonSPEFile3
from .allYouCanEat import AllYouCanEat, normalization,\
    concatenate_data_sets, get_AllYouCanEat_scan, save_data_set,\
    save_frame_mean, get_frame_mean, load_npz_to_Scan
from .ntb import NtbFile

def load_fitarg(fp):
    """load the fit results from fp"""
    from json import load

    with open(fp) as json_data:
        ret = load(json_data)
    return ret
