from os import path
from numpy import genfromtxt, array, delete, zeros, arange,\
    ndarray, concatenate, savez, sqrt, load, ones
from pandas import DataFrame, Series
from copy import deepcopy
from .veronica import PIXEL, SPECS, pixel_to_nm
import warnings
from ..core.scan import Scan

# The meaning of the Indices in the data array
x_pixel_index = -1
y_pixel_index = -2
spec_index = y_pixel_index
frame_axis_index = -3
pp_index = -4 # pump-probe delay

def get_AllYouCanEat_scan(fname, baseline, ir_profile,
                          wavenumber=arange(1600), dir_profile=None, dbaseline=None):
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
    baseline_frames = baseline.data.shape[frame_axis_index]
    data_frames = ret.data.shape[frame_axis_index]

    ret.baseline = baseline.mean(frame_axis_index).squeeze()
    if baseline_frames > 1:
        ret.dbaseline = baseline.std(frame_axis_index).squeeze() / sqrt(baseline_frames-1)
        
    ret.back_sub = ret.data - ret.baseline
    ret.frame_mean = ret.back_sub.mean(frame_axis_index).squeeze()

    # uncertainly of the norm depends of the measurement itself and on
    # the uncertainly of the Baseline as well.
    if data_frames > 1:
        ddata = ret.data.std(frame_axis_index).squeeze() / sqrt(data_frames-1)
        ret.dframe_mean = sqrt(
            (ddata)**2 + (ret.dbaseline)**2
        )
    return ret

def normalization(DataContainer, baseline, ir_profile, dbaseline=None, dir_profile=None):
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
        [elm.data for elm in list_of_data_sets], frame_axis_index
    )
    ret.back_sub = concatenate(
        [elm.back_sub for elm in list_of_data_sets], frame_axis_index
    )
    ret.dback_sub = concatenate(
        [elm.dback_sub for elm in list_of_data_sets], frame_axis_index
    )
    ret.normalized = concatenate(
        [elm.normalized for elm in list_of_data_sets], frame_axis_index
    )
    ret.dnormalized = concatenate(
        [elm.dnormalized for elm in list_of_data_sets], frame_axis_index
    )
    ret.norm = concatenate(
        [elm.norm for elm in list_of_data_sets], frame_axis_index
    )
    ret.dnorm = concatenate(
        [elm.dnorm for elm in list_of_data_sets], frame_axis_index
    )
    ret.dates = concatenate(
        [elm.dates for elm in list_of_data_sets]
    )
    ret.sums = ret.norm.reshape(
        ret.norm.shape[frame_axis_index], ret.norm.shape[x_pixel_index]
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
    frames = data_container.data.shape[frame_axis_index]
    data = data_container.data.mean(frame_axis_index)
    back_sub = data_container.back_sub.mean(frame_axis_index)
    ddata = None
    dback_sub = None
    dnormalized = None

    if frames > 1:
        ddata = data_container.data.std(frame_axis_index) / sqrt(frames - 1)
        dback_sub = data_container.back_sub.std(frame_axis_index) / sqrt(frames - 1)

    normalized = getattr(data_container, 'normalized', None)
    if not isinstance(normalized, type(None)):
        normalized = normalized.mean(frame_axis_index)

    norm = getattr(data_container, 'norm', None)
    if not isinstance(norm, type(None)):
        norm = norm.mean(frame_axis_index)

    base = getattr(data_container, 'base', None)
    if not isinstance(base, type(None)):
        base = base.mean(frame_axis_index)


    if not isinstance(normalized, type(None)) and frames > 1:
        # This is only the statistical fluctuation of the normalized data itself
        dnormalized = data_container.normalized.std(frame_axis_index) / sqrt(frames - 1)
        # we must also add the fluctuations due to the substraction of the baseline
        # and the normalization with the ir profile. These fluctuations
        # Themselves are stored in the dnormalized property of the data_container.
        # If I'm not mistaken we can just add them, because Gaussian error propagation
        # for the a+b is da+db
        dnormalized += data_container.dnormalized.mean(frame_axis_index)

    dnorm = None
    if not isinstance(norm, type(None)) and frames > 1:
        dnorm = data_container.norm.std(frame_axis_index) / sqrt(frames - 1)
        dnorm =+ data_container.dnorm.mean(frame_axis_index)

    dbase = None
    if not isinstance(base, type(None)) and frames > 1:
        dbase = data_container.base.std(frame_axis_index) / sqrt(frames - 1)
        dbase += data_container.dbase.mean(frame_axis_index)

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
    column_names = ['spec_%i' % i for i in range(imp['data'].shape[spec_index])]
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

class AllYouCanEat():
    """Class to quickly import data.

    It is ment to read as many different datastructures as I find
    here at the MPIP.

    Reads
    -----
      - veronica.dat files
      - veronica save files
      - .spe files from Andor version 2.5 and version 3

    Properties
    ----------
    data : 4 dim numpy array
        Each axis seperates the data by:
        - axis 0: pump-probe time delay
        - axis 1: frames
        - axis 2: y-pixels
        - axis 3: x-pixels
    """
    def __init__(self, fname=None):
        self.pp_delays = array([0])
        self.metadata = {}
        self._fname  = fname

        if isinstance(fname, type(None)):
            return

        if '~' in fname:
            fname = path.expanduser(fname)
        self._fname = path.abspath(fname)
        self._dates = None
        self._readData()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._data = value

    @property
    def dates(self):
        if isinstance(self._dates, type(None)):
            self._dates = []
            for i in range(self.data.shape[1]):
                self._dates.append(
                    self.metadata['date'] + i * self.metadata['exposure_time']
                )
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def times(self):
        ret = []
        for i in range(self.data.shape[1]):
            ret.append(
                i * self.metadata['exposure_time']
            )
        return ret

    def _readData(self):
        self._get_type()
        self._arrange_data()
        self._read_metadata()
        self._make_calibration()

    def _get_type(self):
        self._get_type_by_path()
        self._check_type()

    def _arrange_data(self):
        if self._type == "spe":
            self._arrange_spe()
            return

        if self._type == "sp":
            self._arrange_sp()
            return

        if self._type == "ts" or self._type == 'sc':
            self._arrange_ts()
            return

        raise NotImplementedError("Uuuups this should never be reached."
                                  "Bug with %s" % self._fname)

    def _read_metadata(self):
        """Read metadata of the file """

        from ..utils.metadata import get_metadata_from_filename

        try:
            metadata = get_metadata_from_filename(self._fname)
        except ValueError:
            warnings.warn('ValueError while trying to extract metadata from filepath./nSkipping')
            return

        if self._type == 'spe':
            # At first we use the metadata entries from the file
            # content. They are loaded during the arrange step.
            # Here we add what we can get from the filename.
            # file content metadata wins over
            # filename metadata
            # self.metadata = {**self.metadata, **metadata}
            ret = metadata.copy()
            ret.update(self.metadata.copy())
            self.metadata = ret
            return

        self.metadata = metadata

    def _get_type_by_path(self):
        """Get datatype by looking at the path.

        Sets self._type to a string describing the data. And returns true
        at the first success. Returns False if type could not be deduced
        by name."""
        # First check the name. Maybe it follows my naming convention.
        fhead, ftail = path.split(self._fname)

        if path.splitext(ftail)[1] == '.spe':
            self._type = 'spe'
            return

        if ftail.find('_sp_') != -1:
            self._type = 'sp'
            return True

        if ftail.find('_sc_') != -1:
            self._type = 'sc'
            return True

        if ftail.find('_ts_') != -1:
            self._type = 'ts'
            return True

        return False

    def _check_type(self):
        """Check the type by comparing name and data shape."""

        # .spe is binary and hard to check by shapre.
        # We will assume it was not named wrongly and
        # just go on.
        if self._type == 'spe':
            return True

        typeByName = self._type
        self._data = genfromtxt(self._fname)
        typeByShape = self._get_type_by_shape()

        if typeByName == typeByShape:
            return True

        print("type by name and type by shape don't match in %s" % self._fname)
        print("typeByName : " + typeByName + " typeByShape: " + typeByShape)
        print("Fallback to type by shape.")
        return False

    def _get_type_by_shape(self):
        """Get type from the shape of data."""
        shape = self._data.shape

        if shape == (1600, 6):
            return 'sp'

        # If shape is not correct raise an error. This must
        # be handeled. It means another program is used, or data
        # is possible corrupted
        if shape[0] % 1602 != 0 or shape[1] % 6 != 0:
            raise IOError("Can't understand shape of data in %s" % self._fname)

        # The number of repetitions
        # -1 because the AVG is saved into this file as well
        self.NumReps = self._data.shape[1] // 6 - 1

        # The numner of pp_delays
        self.NumPp_delays = self._data.shape[0] // 1602

        if self.NumPp_delays == 1:
            return 'sc'

        if self.NumPp_delays > 1:
            return 'ts'

        raise NotImplementedError(
            "Uuups this should never be reached."
            "Shape of %s is %s" % (self._fname, self._data.shape)
        )

    def _arrange_spe(self):
        from .spe import PrincetonSPEFile3

        sp = PrincetonSPEFile3(self._fname)
        self._data = sp.data.reshape(1, sp.NumFrames, sp.ydim, sp.xdim)
        self.metadata['central_wl'] = sp.central_wl
        self.metadata['exposure_time'] = sp.exposureTime
        self.metadata['gain'] = sp.gain
        self.metadata['sp_type'] = 'spe'
        self.metadata['date'] = sp.date
        self.metadata['tempSet'] = sp.tempSet
        self.wavelength = sp.wavelength
        self.calib_poly = sp.calib_poly
        self.pixel = arange(sp.xdim)

    def _arrange_sp(self):
        """Makes a scan having the same data structure as spe """
        # 3 because victor saves 3 spectra at a time
        # 1 because its only 1 spectrum no repetition
        # x axis given by the pixels
        self._data = self._data[:, 1:4].reshape(1, 1, 3, PIXEL)

    def _arrange_ts(self):
        self._data

        def _iterator(elm):
            return array(
                # ditch first iteration because it the AVG
                # and correct NumReps for the missing AVG entry
                [elm(i) for i in range(1, self.NumReps + 1)]
            ).flatten().tolist()

        # Remove unneeded columns
        colum_inds = [0] + _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
        self._data = self._data[:, colum_inds]

        # pop pp_delays from data array and remove rows
        self.pp_delays = self._data[0::1602, 0]
        self._data = delete(self._data, slice(0, None, 1602), 0)

        # Now We know the rest is uint32
        self._data = self._data.astype('uint32')

        # pop empty rows
        empty_rows = [i*1600 + i-1 for i in range(self._data.shape[0]//1600)]
        self._data = delete(self._data, empty_rows, 0)

        # Currently numpy ignores the [-1, ...] in the empty_rows, but we want
        # the last line to be deleted as well, because it is empty
        if self._data.shape[0] % 1600 == 1:
            self._data = delete(self._data, -1, 0)

        # Remove the 0 (pixel) column it is redundant
        self._data = self._data[:, 1:]

        # We want data to fit in the same structure as all the other scans
        ret = zeros(
            (self.NumPp_delays, self.NumReps, SPECS, PIXEL)
        )
        for i_row in range(len(self._data)):
            row = self._data[i_row]
            # The index of the current delay
            i_delay = i_row // PIXEL
            i_pixel = i_row % PIXEL
            for i_col in range(len(row)):
                # The index of the current repetition
                i_rep = i_col // self.NumReps
                i_spec = i_col % SPECS  # 3 is the number of specs
                ret[i_delay, i_rep, i_spec, i_pixel] = row[i_col]

        self._data = ret

    def _make_calibration(self):
        from ..utils.static import nm_to_ir_wavenumbers

        if self._type != 'spe':
            self.pixel = arange(PIXEL)
            cw = self.metadata.get('central_wl')
            if cw and cw >= 1:
                self.wavelength = pixel_to_nm(
                    arange(PIXEL),
                    self.metadata.get('central_wl')
                )
        # wavelength does not exist when central wl is wrong in
        # filename in that case we will use pixel so wavenumber
        # is not empty
        wavelength = getattr(self, "wavelength", self.pixel)
        self.wavenumber = nm_to_ir_wavenumbers(
            wavelength, self.metadata.get('vis_wl', 810)
        )
