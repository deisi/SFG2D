# REFACTOR, I think the complete module is trash
from os import path
from numpy import genfromtxt, array, delete, zeros, arange,\
    ndarray, concatenate, savez, sqrt, load, ones, empty, dtype
from pandas import DataFrame, Series
from copy import deepcopy
import warnings
from ..core import SfgRecord
from ..core.scan import Scan
from .. utils.consts import X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX, FRAME_AXIS_INDEX, PIXEL

debug=0
#TODO Refactor all these functions. They are too specific.
def get_AllYouCanEat_scan(fname, baseline, ir_profile,
                          wavenumber=arange(PIXEL), dir_profile=None, dbaseline=None):
    '''Function to unify usual data processing pipeline.

    This function imports data, subtracts the baseline and normalizes the data.
    fname: str
        file to load
    baseline: array
        baseline data
    ir_profile: array
        normalization data
    wavenumber: array
        wavenumbers of this scan. Must be same as for baseline and ir_profile
    dir_profile: array
        uncertainty of  the normalization
    dbaseline: array
        uncertainty of the baseline

    '''
    ret = AllYouCanEat(fname)
    ret.baseline = baseline
    ret = normalization(ret, baseline, ir_profile, dbaseline, dir_profile)
    ret.wavenumber = wavenumber

    return ret

def get_frame_mean(fname, fbaseline):
    ret = AllYouCanEat(fname)
    baseline = AllYouCanEat(fbaseline).data
    baseline_frames = baseline.data.shape[FRAME_AXIS_INDEX]
    data_frames = ret.data.shape[FRAME_AXIS_INDEX]

    ret.baseline = baseline.mean(FRAME_AXIS_INDEX).squeeze()
    if baseline_frames > 1:
        ret.dbaseline = baseline.std(FRAME_AXIS_INDEX).squeeze() / sqrt(baseline_frames)

    ret.back_sub = ret.data - ret.baseline
    ret.frame_mean = ret.back_sub.mean(FRAME_AXIS_INDEX).squeeze()

    # uncertainly of the norm depends of the measurement itself and on
    # the uncertainly of the Baseline as well.
    if data_frames >= 1:
        ddata = ret.data.std(FRAME_AXIS_INDEX).squeeze() / sqrt(data_frames)
        ret.dframe_mean = sqrt(
            (ddata)**2 + (ret.dbaseline)**2
        )
    return ret

def normalization(DataContainer, baseline,
                  ir_profile, dbaseline=None, dir_profile=None):
    '''Function to add baseline substraction and normalization

    The DataContainer get added a base_sub property, that
    holds the baseline substracted data. Secondly a
    norm property is added, that encapsulates the data after
    normalization'''

    data_shape = DataContainer.data.shape

    if isinstance(dir_profile, type(None)):
        dir_profile = zeros(data_shape)
    if isinstance(dbaseline, type(None)):
        dbaseline = zeros(data_shape)

    DataContainer.back_sub = DataContainer.data - baseline
    # Up to now the uncertainty of the data is only given by the
    # uncertainty of the baseline, because data is in general
    # not averaged at this point.
    DataContainer.dback_sub = dbaseline * ones(data_shape)

    # Add the normalization spectra
    DataContainer.norm = ir_profile * ones(data_shape)
    DataContainer.dnorm = dir_profile * ones(data_shape)

    DataContainer.normalized = DataContainer.back_sub / ir_profile
    # This uncertainty is given by the
    # ir_profile and the baseline subtracted spectrum
    DataContainer.dnormalized = sqrt(
        (DataContainer.back_sub / (ir_profile)**2 * dir_profile)**2 +\
        (DataContainer.dback_sub / ir_profile)**2
    )

    # add the baseline
    DataContainer.base = baseline * ones(data_shape)
    # They are the same but for sake of completeness I add them here
    DataContainer.dbase = DataContainer.dback_sub
    return DataContainer


def concatenate_data_sets(
        list_of_data_sets,
        sum_sl=slice(None, None)):
    '''Concatenate different measurements.

    list_of_data_sets: array type
        list for the different data sets. Each element must have
        Members as
            - data
              The raw data
            - back_sub
              The data after baseline subtraction
            - norm
              The data after normalization
            - dates
              The date when the data was recorded

    sum_sl : slice
        The slice that is used to calculate the sums of the
        individual spectra of. Default is to sum the whole
        spectra.
    '''

    ret = deepcopy(list_of_data_sets[0])
    ret.data = concatenate(
        [elm.data for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.back_sub = concatenate(
        [elm.back_sub for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.dback_sub = concatenate(
        [elm.dback_sub for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.normalized = concatenate(
        [elm.normalized for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.dnormalized = concatenate(
        [elm.dnormalized for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.norm = concatenate(
        [elm.norm for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.dnorm = concatenate(
        [elm.dnorm for elm in list_of_data_sets], FRAME_AXIS_INDEX
    )
    ret.dates = concatenate(
        [elm.dates for elm in list_of_data_sets]
    )
    ret.sums = ret.norm.reshape(
        ret.norm.shape[FRAME_AXIS_INDEX], ret.norm.shape[X_PIXEL_INDEX]
    )[:, sum_sl].sum(1)


    # add readable dates that can be used as labels
    ret.l_dates = [elm.strftime("%H:%M") for elm in ret.dates]
    # add readable times that can be used as labels
    ret.l_times = [time.seconds//60 for time in ret.times]

    return ret

def save_data_set(fname, data_container):
    """Save a data set as .npz binary file.

    All attributes are save optionally. If missing, None is saved
    Keyword Arguments:
    data_container --
    """
    if '~' in fname:
        fname = path.expanduser(fname)

    # Optional attributes
    wavelength = getattr(data_container, "wavelength", None)
    wavenumber = getattr(data_container, 'wavenumber', None)
    back_sub = getattr(data_container, 'back_sub', None)
    l_dates = getattr(data_container, "l_dates", None)
    times = getattr(data_container, "times", None)
    l_times = getattr(data_container, "l_times", None)
    dnormalized = getattr(data_container, "dnormalized", None)
    normalized = getattr(data_container, 'normalized', None)
    vis_wl = getattr(data_container, 'vis_wl', None)
    metadata = getattr(data_container, 'metadata', None)

    savez(
        fname,
        wavelength=wavelength,
        wavenumber=wavenumber,
        back_sub=back_sub,
        normalized=normalized,
        dnormalized=dnormalized,
        metadata=metadata,
        l_dates=l_dates,
        times=times,
        l_times=l_times,
        vis_wl = vis_wl,
    )

def save_frame_mean(fname, data_container):
    '''Saves a version with only the mean results.

    Also calculates std. errors for the averaged spectra if possible
    '''
    if '~' in fname:
        fname = path.expanduser(fname)

    # load attributes if available
    wavelength = getattr(data_container, "wavelength", None)
    wavenumber = getattr(data_container, 'wavenumber', None)
    times = getattr(data_container, "times", None)
    metadata = getattr(data_container, 'metadata', None)
    # number of frames (usually = number of repetitions)
    frames = data_container.data.shape[FRAME_AXIS_INDEX]
    data = data_container.data.mean(FRAME_AXIS_INDEX)
    back_sub = data_container.back_sub.mean(FRAME_AXIS_INDEX)
    ddata = None
    dback_sub = None
    dnormalized = None

    if frames >= 1:
        ddata = data_container.data.std(FRAME_AXIS_INDEX) / sqrt(frames)
        dback_sub = data_container.back_sub.std(FRAME_AXIS_INDEX) / sqrt(frames)

    normalized = getattr(data_container, 'normalized', None)
    if not isinstance(normalized, type(None)):
        normalized = normalized.mean(FRAME_AXIS_INDEX)

    norm = getattr(data_container, 'norm', None)
    if not isinstance(norm, type(None)):
        norm = norm.mean(FRAME_AXIS_INDEX)

    base = getattr(data_container, 'base', None)
    if not isinstance(base, type(None)):
        base = base.mean(FRAME_AXIS_INDEX)

    if not isinstance(normalized, type(None)) and frames > 1:
        # This is only the statistical fluctuation of the normalized data itself
        dnormalized = data_container.normalized.std(FRAME_AXIS_INDEX) / sqrt(frames)
        # we must also add the fluctuations due to the substraction of the baseline
        # and the normalization with the ir profile. These fluctuations
        # Themselves are stored in the dnormalized property of the data_container.
        # If I'm not mistaken we can just add them, because Gaussian error propagation
        # for the a+b is da+db
        dnormalized += data_container.dnormalized.mean(FRAME_AXIS_INDEX)

    dnorm = None
    if not isinstance(norm, type(None)) and frames >= 1:
        dnorm = data_container.norm.std(FRAME_AXIS_INDEX) / sqrt(frames)
        dnorm =+ data_container.dnorm.mean(FRAME_AXIS_INDEX)

    dbase = None
    if not isinstance(base, type(None)) and frames >= 1:
        dbase = data_container.base.std(FRAME_AXIS_INDEX) / sqrt(frames)
        dbase += data_container.dbase.mean(FRAME_AXIS_INDEX)

    savez(
        fname,
        wavelength=wavelength,
        wavenumber=wavenumber, # wavenumbers
        data=data, # raw data
        ddata=ddata, # uncertaincy of the raw data
        back_sub=back_sub, # baseline substracted data
        dback_sub=dback_sub, # uncertaincy of the baseline substracted data
        normalized=normalized, # normalized data
        dnormalized=dnormalized,  # uncertaincy of the normalized data
        norm=norm, # ir profile
        dnorm=dnorm, # uncertainly of the ir profile
        base=base, # baseline
        dbase=dbase,
        times=times,
        metadata=metadata,
    )


def load_npz_to_Scan(fname, **kwargs):
    """Translates an AllYouCanEat obj into a Scan obj.

    Works only for 2d squeezable data."""
    if '~' in fname:
        fname = path.expanduser(fname)

    imp = load(fname)

    # TODO make this ready for 2d input
    ret = Scan()
    column_names = ['spec_%i' % i for i in range(imp['data'].shape[SPEC_INDEX])]
    ret.df = DataFrame(
        imp['data'].squeeze().T,
        index=imp['wavenumber'].squeeze(),
        columns = column_names,
        **kwargs
    )
    ret.base = imp['base'].squeeze()
    ret.dbase = imp['dbase'].squeeze()
    ret.norm = imp['norm'].squeeze()
    ret.dnorm = imp['dnorm'].squeeze()
    ret.metadata = imp['metadata'].squeeze()[()]
    ret.df['dspec_0'] = imp['ddata'].squeeze()
    ret.df.index.name = 'wavenumber'
    ret.normalized = imp['normalized'].squeeze()
    ret.dnormalized = imp['dnormalized'].squeeze()

    # if all arrays are 2d squeezable and have the same shape,
    # combine them all in df, so all data is in the same pandas
    # data frame
    for elm in (ret.base, ret.dbase, ret.norm, ret.dnorm):
        test_shapes = elm.shape[0] == ret.df.shape[0]
        if not test_shapes:
            return ret

    # must flip everything if sorted wrongly
    if imp['wavenumber'][0] - imp['wavenumber'][-1] > 0:
        ret.df.sort_index(inplace=True)
        ret.norm = ret.norm[::-1]
        ret.dnorm = ret.dnorm[::-1]
        ret.base = ret.base[::-1]
        ret.dbase = ret.dbase[::-1]
        ret.normalized = ret.normalized[::-1]
        ret.dnormalized = ret.dnormalized[::-1]

    ret.df['norm'] = ret.norm
    ret.df['dnorm'] = ret.dnorm
    ret.df['base'] = ret.base
    ret.df['dbase'] = ret.dbase
    ret.df['normalized'] = ret.normalized
    ret.df['dnormalized'] = ret.dnormalized
    return ret

#TODO Rename to SFGData
# This is here for backwards compactibility reasons
# The name and the location of the class was bad. It sould
# not be used any more

# DEPRECATED
class AllYouCanEat(SfgRecord):
    pass
