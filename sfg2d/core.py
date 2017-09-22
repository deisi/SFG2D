from os import path
import warnings

import numpy as np
from scipy.signal import medfilt
from scipy.stats import sem

from .io.veronica import pixel_to_nm, get_from_veronika
from .io.victor_controller import get_from_victor_controller
from .utils import (
    nm_to_ir_wavenumbers, nm_to_wavenumbers, X_PIXEL_INDEX, Y_PIXEL_INDEX,
    FRAME_AXIS_INDEX, PIXEL, PP_INDEX, SPEC_INDEX, find_nearest_index
)
from .utils.consts import (VIS_WL, PUMP_FREQ, NORM_SPEC, BASE_SPEC,
                           FRAME_AXIS_INDEX_P,  PP_INDEX_P)
from .utils.filter import double_resample


class SfgRecord():
    """Class to load and manage SFG data in a 4D structure

    It is ment to encapsulate raw SFG data in a unifying manner.
    Four dimensions are needed to cope with pump-probe delay,
    frames(repetitions), y-pixel and x-pixel.

    Reads
    -----
      - veronica .dat files
      - .spe files from Andor version 2.5 and version 3
      - victor_controller .dat files

    Parameters
    ----------
    fname: string
        Path to input data
    base: 4d castable array.
        Data for the baseline
    norm: 4d castable array
        Data to normalize with

    Properties
    ----------
    raw_data: 4 dim numpy array
        Immutable version of the imported data. This should only be changed
        upon initialization of the data. Else this should always stay the same.
    data : 4 dim numpy array
        Each axis seperates the data by:
        - axis 0: pump-probe time delay
        - axis 1: frames
        - axis 2: y-pixels
        - axis 3: x-pixels

    dates: list of dates
        eaclist is a relative timedelta obj, that refers
        to the time 0 of the first frame.
    frames: int
        Number of frames of this obj.
    pixel: np.array
        numpy array with the range of the pixels. Often used as x-axis.
    wavelength: np.array
        Wavelength of the data.
            If data is from an .spe file, wavelength can be read from it.
            If data is from a ,dat file, wavelength gets calcuated based
              on a calibration line with the central wavelength taken from:
              `SfgRecord.metadata['central_wl']`
    wavenumber: np.array
        Wavenumber calculated from the wavelength as described above.
        The effect of the up-conversion is calculated by using
        `SfgRecord.metadata["vis_wl"]` as wavelength of the visible.
    basesubed: np.array
        4 d baselinesubstracted array
    normalized: np.array
        4 d normalized array
    """
    def __init__(self, fname=None, rawData=np.zeros((1, 1, 1, PIXEL)),
                 base=None, norm=None, baseline_offset=0):

        ## Beacaue I know it will happen and we cans safely deal with it.
        #np.seterr(divide='ignore')
        #np.seterr(invalid='ignore')

        # 1d Array of pump-probe delay values
        self.pp_delays = np.array([0])

        # Dicitionary of metadata
        self.metadata = {}

        # File name of the data file
        if not fname:
            self._fname = ""
        else:
            self._fname = fname

        # 4d array of raw data
        self._rawData = rawData

        # Error of raw data
        self._rawDataE = None

        # 4d buffer array of processed data
        self._data = None

        # 4d array of baseline/background data
        self._base = None

        # Constant offset for baseline
        self._baseline_offset = baseline_offset

        # error of base
        self._baseE = None

        # 4d array of normalized data
        self._norm = None

        # error of normalized data
        self._normE = None

        # Boolean to check baseline subtraction
        self.isBaselineSubed = False

        # 4d array of baselinesubstracted data
        self._basesubed = None

        # Error ob  baselinesubtracted data
        self._basesubedE = None

        # boolean for normalization
        self.isNormalized = False

        # 4d array of normalized data
        self._normalized = None

        # error of normalized data
        self._normalizedE = None

        # type of data file
        self._type = 'unknown'

        # 1d array with wavelength values
        self._wavelength = None
        self._setted_wavelength = False

        # 1d array with wavenumber values
        self._wavenumber = None
        self._setted_wavenumber = False

        # y-pixel/spectra index of pumped data
        self._pumped_index = 0

        # index of unpumped data
        self._unpumped_index = 1

        # List of dates the spectra were recorded at.
        self._dates = None

        # Boolean to toggle default zero_time subtraction
        self._zero_time_subtraction = True

        # list/slice of delay indexes for zero_time_subtraction
        self._zero_time_selec = [0]

        # array of bleach value at negative time
        self.zero_time_abs = None

        # array of relative bleach value at negative time.
        self.zero_time_rel = None

        # array of normalized bleach at negative time.
        self.zero_time_norm = None

        # 4d array of values for the static drift correction.
        self._static_corr = None

        # Region of interest for the x_spec
        self.roi_x_pixel_spec = slice(0, PIXEL)

        # Region of interest x_pixel
        self.rois_x_pixel_trace = [slice(0, PIXEL)]

        # Region of interest y_pixel/spectra
        self.roi_spectra = slice(0, None)

        # Region of interest frames
        self.roi_frames = slice(0, None)

        # Region of interest pp_delays
        self.roi_delay = slice(0, None)

        # Subreions of interest for pump probe
        self.rois_delays_pump_probe = [slice(0, None)]

        # Buffer for traces. kwg is binning and then [X, Y, Yerr]
        self.traces = {}

        # Buffer for fot models.
        self.models = {}

        # A short name for the record
        self.name = ''

        # A Nice Latex Version of the name of the recrod
        self.lname = ''

        if isinstance(fname, type(None)):
            return

        # Allows the use of ~ on windows and linux
        if '~' in fname:
            fname = path.expanduser(fname)
        self._fname = path.abspath(fname)
        self._dates = None

        self._readData()

        if isinstance(base, type(None)):
            base = BASE_SPEC
        else:
            if isinstance(base, str):
                base = SfgRecord(base).data
                self.base = base
            else:
                self.base = base

        if isinstance(norm, type(None)):
            norm = NORM_SPEC
        else:
            if isinstance(norm, str):
                norm = SfgRecord(norm).data
                self.norm = norm
            else:
                self.norm = norm

    def __repr__(self):
        msg = '# ppelays: {}\n'.format(self.number_of_pp_delays)
        msg += '# frames: {}\n'.format(self.number_of_frames)
        msg += '# y-pixel: {}\n'.format(self.number_of_y_pixel)
        msg += '# x-pixel: {}\n'.format(self.number_of_x_pixel)
        msg += 'Wavelength {:.0f} to {:.0f}\n'.format(self.wavelength.min(),
                                              self.wavelength.max())
        msg += 'Wavenumber: {:.0f} to {:.0f}\n'.format(self.wavenumber.min(),
                                               self.wavenumber.max())
        msg += 'Baseline @ {}\n'.format(self.base.mean((0, 1, 3)))
        msg += 'Norm with {}\n'.format(self.norm.mean((0, 1, 3)))
        msg += 'Metadata is {}\n'.format(self.metadata)
        return msg

    @property
    def saveable(self):
        """A dict of the saveable properties.

        This dict translates between the propertie names and how they
        are saved in an npz."""

        # This is needed, because one might or might not circumvent the setter
        # and getter functions of the properties.
        saveable = {}
        saveable['rawData'] = '_rawData'
        saveable['metadata'] = 'metadata'
        saveable['norm'] = '_norm'
        saveable['base'] = '_base'
        saveable['pp_delays'] = 'pp_delays'
        #saveable['wavelength'] = '_wavelength'
        #saveable['wavenumber'] = '_wavenumber'
        #saveable['dates'] = '_dates'
        saveable['pumped_index'] = '_pumped_index'
        saveable['unpumped_index'] = '_unpumped_index'
        #saveable['rawDataE'] = '_rawDataE'
        #saveable['normE'] = '_normE'
        #saveable['baseE'] = '_baseE'
        #saveable['basesubedE'] = '_basesubedE'
        #saveable['normalizedE'] = 'normalizedE'
        saveable['baseline_offset'] = '_baseline_offset'
        saveable['zero_time_subtraction'] = '_zero_time_subtraction'
        saveable['zero_time_selec'] = 'zero_time_selec'
        saveable['rois_x_pixel_trace'] = 'rois_x_pixel_trace'
        saveable['roi_x_pixel_spec'] = 'roi_x_pixel_spec'
        saveable['roi_spectra'] = 'roi_spectra'
        saveable['roi_frames'] = 'roi_frames'
        saveable['roi_delay'] = 'roi_delay'
        saveable['rois_delays_pump_probe'] = 'rois_delays_pump_probe'
        return saveable

    def select(
            self,
            prop="normalized",
            prop_kwgs={},
            roi_delay=None,
            roi_frames=None,
            roi_spectra=None,
            roi_pixel=None,
            frame_med=False,
            delay_mean=False,
            spectra_mean=False,
            pixel_mean=False,
            medfilt_pixel=1,
            resample_freqs=0
    ):
        """Central Interface to select data.

        prop: Name of the propertie to select data from
        prop_kwgs: For Infered properties, spectial kwgs,
        roi_delay: delay slice
        ...
        """
        if isinstance(roi_delay, type(None)):
            roi_delay = self.roi_delay
        if isinstance(roi_frames, type(None)):
            roi_frames = self.roi_frames
        if isinstance(roi_spectra, type(None)):
            only_one_spec = ('bleach', 'pumped', 'unpumped')
            if any([teststr in prop for teststr in only_one_spec]):
                roi_spectra = [0]
            else:
                roi_spectra = self.roi_spectra
        if isinstance(roi_pixel, type(None)):
            roi_pixel = self.roi_x_pixel_spec
        # Usually X-Axis properties.
        if prop in ('pixel', 'wavenumber', 'wavelength', 'pp_delays', 'frames'):
            ret = getattr(self, prop)
            if prop in ('pixel', 'wavenumber', 'wavelength'):
                ret = ret[roi_pixel]
            elif prop is "pp_delays":
                ret = ret[roi_delay]
            elif prop is "frames":
                ret = ret[roi_frames]
            return ret

        # Real properties get used directly
        if prop in ('rawData', 'basesubed', 'normalized', 'base', 'norm'):

            ret = getattr(self, prop)
        # Infered properites need aditional kwgs
        elif prop in ('pumped', 'unpumped', 'bleach', 'trace'):
            ret = getattr(self, prop)(**prop_kwgs)
        else:
            raise NotImplementedError('{} not know'.format(prop))

        ret = ret[
                roi_delay,
                roi_frames,
                roi_spectra,
                roi_pixel,
               ]
        # Frame first to remove spikes.
        if frame_med:
            ret = np.median(ret, FRAME_AXIS_INDEX, keepdims=True)
        if delay_mean:
            ret = np.mean(ret, PP_INDEX, keepdims=True)
        if spectra_mean:
            ret = np.mean(ret, SPEC_INDEX, keepdims=True)
        if medfilt_pixel > 1:
            ret = medfilt(ret, (1, 1, 1, medfilt_pixel))
        if resample_freqs:
            ret = double_resample(ret, resample_freqs, X_PIXEL_INDEX)
        # Median because sometimes there are still nan and infs left.
        if pixel_mean:
            ret = np.median(ret, X_PIXEL_INDEX, keepdims=True)
        return ret

    def sem(self, prop, **kwgs):
        """Returns standard error of the mean of given property.

        kwgs: get passed to SfgRecord.subselect
        """
        #Forced because error is calculated over the frame axis.
        kwgs['frame_med'] = False
        kwgs['prop'] = prop
        if 'trace' in prop:
            kwgs['pixel_mean'] = True
        return sem(self.select(**kwgs), FRAME_AXIS_INDEX)

    @property
    def rawData(self):
        """Raw Data of the Record.

        `SfgRecord.rawData` is a 4d array with:
        0 axis: pp_delay index.
        1 axis: frame index.
        2 axis: y_pixel/spectra index.
        3 axis: pixel index."""
        return self._rawData

    @rawData.setter
    def rawData(self, value):
        """Raw Data of the Record"""
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._rawData = value

        # Must reset all deduced properties.
        self._data = self._rawData
        self._basesubed = None
        self._normalized = None
        self.isBaselineSubed = False
        self.isNormalized = False

    @property
    def base(self):
        """Baseline/Background data.

        4d data array with the same structure as `SfgRecord.rawData`."""
        if isinstance(self._base, type(None)):
            ret = np.zeros_like(self.rawData) + self.baseline_offset
            return ret
        ret = self._base + self.baseline_offset
        return ret

    @base.setter
    def base(self, value):
        if self.isBaselineSubed:
            self.add_base()
            self.isBaselineSubed = False
        self._base = value * np.ones_like(self.rawData)
        # Reset the dependent properties
        self._basesubed = None
        self._normalized = None

    @property
    def norm(self):
        """Spectrum for normalization.

        The quartz or gold or what ever spectrum that is your reference.
        same structure as `SfgRecord.rawData`."""
        ret = self._norm
        if isinstance(ret, type(None)):
            ret = np.ones_like(self.rawData)
        return ret

    @norm.setter
    def norm(self, value):
        if self.isNormalized:
            self.un_normalize()
            self.isNormalized = False
        # Make 0 values very small so division is not a problem.
        value[np.isin(value, 0)] = 1e-08
        self._norm = value * np.ones_like(self.rawData)
        # Reset dependent properties
        self._normalized = None
        self._pumped_norm = None
        self._unpumped_norm = None

    @property
    def basesubed(self):
        """Baselinesubstracted data.

        The rawData after baseline subtraction."""

        if isinstance(self._basesubed, type(None)):
            self._basesubed = self.rawData - self.base
        return self._basesubed

    @property
    def normalized(self):
        """Normalized data.

        4D array like `SfgRecord.rawData` after normalizting.
        """
        if isinstance(self._normalized, type(None)):
            self._normalized = self.basesubed / self.norm
            number_of_nan_infs = np.count_nonzero(
                np.isnan(self._normalized) | np.isinf(self._normalized)
               )
            self.normalized_nan_infs = number_of_nan_infs
            np.nan_to_num(self._normalized, copy=False)

        return self._normalized

    @property
    def baseline_offset(self):
        """Constant factor to add to the baseline."""
        return self._baseline_offset

    @baseline_offset.setter
    def baseline_offset(self, value):
        self._baseline_offset = value
        # Reset the dependent properties
        self._basesubed = None
        self._normalized = None

    @property
    def rawDataE(self):
        """Std error of the mean rawData.
        """
        self._rawDataE = self._calc_sem('rawData')
        return self._rawDataE

    @property
    def baseE(self):
        """Error of the baseline/background.
        """
        self._baseE = self._calc_sem('base')
        return self._baseE

    @property
    def normE(self):
        """Error of the normalization spectrum.

        Same structure as `SfgRecord.rawData`."""
        self._normE = self._calc_sem('norm')
        return self._normE

    @property
    def basesubedE(self):
        """Data error after subtracting baseline."""
        self._basesubedE = self._calc_sem('basesubed')
        return self._basesubedE

    @property
    def normalizedE(self):
        """Error of the normalized data."""
        self._normalizedE = self._calc_sem('normalized')
        return self._normalizedE

    @property
    def exposure_time(self):
        return self.metadata.get("exposure_time")

    @property
    def exposure_time_ms(self):
        """Exposure time in microseconds for convenience."""
        return int(self.metadata.get("exposure_time").microseconds/10**3)

    @property
    def exporsure_time_s(self):
        return self.metadata.get("exposure_time").seconds

    @property
    def number_of_frames(self):
        """Number of frames"""
        return self.rawData.shape[FRAME_AXIS_INDEX]

    @property
    def pixel(self):
        """Iterable list of pixels."""
        return np.arange(self.rawData.shape[X_PIXEL_INDEX])

    @property
    def number_of_y_pixel(self):
        """Number of y_pixels/spectra."""
        return self.rawData.shape[Y_PIXEL_INDEX]

    @property
    def number_of_spectra(self):
        """Number of spectra/y_pixels."""
        return self.number_of_y_pixel

    @property
    def number_of_x_pixel(self):
        """Number of pixels."""
        return self.rawData.shape[X_PIXEL_INDEX]

    @property
    def number_of_pp_delays(self):
        """Number of pump probe time delays."""
        return self.rawData.shape[PP_INDEX]

    @property
    def dates(self):
        """List of datetimes the spectra were recorded at.

        A list of datetime objects when each spectrum was created."""
        self._dates = np.arange(
                (self.number_of_pp_delays * self.number_of_frames)
            )
        date = self.metadata.get("date")
        exposure_time = self.metadata.get("exposure_time")
        if date and exposure_time:
            self._dates = date + self._dates * exposure_time
        elif exposure_time:
            self._dates = self._dates * exposure_time
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def frames(self):
        """Array of frames"""
        return np.arange(self.number_of_frames)

    @property
    def times(self):
        """List of timedeltas the spectra were recorded with."""
        ret = []
        time_of_a_scan = self.metadata['exposure_time']
        for i in range(self.rawData.shape[1]):
            for j in range(self.rawData.shape[0]):
                ret.append(
                    (i * self.number_of_pp_delays + j) * time_of_a_scan
                )
        return ret

    @property
    def central_wl(self):
        """Central wavelength of the grating in nm."""
        return self.metadata.get("central_wl", 1)

    @central_wl.setter
    def central_wl(self, value):
        self.metadata["central_wl"] = value

        # For spe files we must rely on the stored
        # wavelength because  we cannot recalculate,
        # since calibration parameters are not always
        # known. From spe version 3 on they are amongst
        # the metadata, but that is not taken into account
        # here.
        if self.metadata.get("sp_type"):
            return

    @property
    def wavelength(self):
        """Numpy array with the wavelength in nm.

        A 1d numpy array with the wavelength of the spectra.
        If wavelength is not set externally, the central wavelength
        is read from *SfgRecord.metadata* and an internaly stored
        calibration line is used to calculate the wavelength.

        Examples
        --------
        If you want to get the automatically obtained wavelength call:
        >>> SfgRecord.wavelength

        To set a specific central wavelength and to recalculate the
        wavelength do:
        >>> SfgRecord.metadata['central_wl'] = 670
        >>> SfgRecord._wavelength = None
        >>> SfgRecord.wavelength

        The above example sets 800 nm as the central wavelength, and forced
        a recalculation of the wavelength data.
        """
        if self._setted_wavelength:
            return self._wavelength
        cw = self.central_wl
        self._wavelength = pixel_to_nm(self.pixel, cw)
        return self._wavelength

    @wavelength.setter
    def wavelength(self, arg):
        print("Set Wavelength Called")
        self._setted_wavelength = True
        self._wavelength = arg

    @property
    def vis_wl(self):
        """Wavelength of the visible in nm."""
        return self.metadata.get("vis_wl", 800)

    @vis_wl.setter
    def vis_wl(self, value):
        self.metadata['vis_wl'] = value

    @property
    def pump_freq(self):
        return self.metadata.get('pump_freq')

    @pump_freq.setter
    def pump_freq(self, value):
        self.metadata['pump_freq'] = value

    @property
    def wavenumber(self):
        """Numpy array of wavenumbers in 1/cm.

        Works similar to *SfgRecord.wavelength*, so see its documentation
        for further insights.
        1d numpy array of wavenumber values in 1/cm.
        First wavelength in nm is calculated using *SfgRecord.wavelength*
        then, wavenumber is calculated using the vis_wl keyword from
        the metadata dictionary.

        Examples
        --------
        Obtaining wavenumber with:
        >>> SfgRecord.wavenumber
        Recalculate with:
        >>> SfgRecord.metadata["vis_wl"]=800
        >>> SfgRecord.metadata["central_wl"]=670
        >>> SfgRecord._wavenumber = None
        >>> SfgRecod._wavelength = None
        >>> SfgRecord.wavenumber
        """

        if self._setted_wavenumber:
            return self._wavenumber
        vis_wl = self.metadata.get("vis_wl")
        if isinstance(vis_wl, type(None)) or vis_wl < 1:
            self._wavenumber = nm_to_wavenumbers(self.wavelength)
        else:
            self._wavenumber = nm_to_ir_wavenumbers(self.wavelength, vis_wl)
        self._wavenumber = np.nan_to_num(self._wavenumber)
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, arg):
        """Setter for wavenumber propertie."""
        self._setted_wavenumber = True
        self._wavenumber = arg

    @property
    def pp_delays_ps(self):
        """Time of pp_delay in ps."""
        return self.pp_delays/1000

    @property
    def zero_time_subtraction(self):
        return self._zero_time_subtraction

    @zero_time_subtraction.setter
    def zero_time_subtraction(self, value):
        self._zero_time_subtraction = value

    @property
    def zero_time_selec(self):
        """Slice/List to select delays for zero time subtraction."""
        return self._zero_time_selec

    @zero_time_selec.setter
    def zero_time_selec(self, value):
        self._zero_time_selec = value

    def wavenumbers2index(self, wavenumbers, sort=False):
        """Calculate index positions of wavenumbers.

        Tries to find matching index values for given wavenumbers.
        The wavenumbers dont need to be exact. Closest match will
        be used.

        Parameters
        ----------
        wavenumbers: iterable
            list of wavenumbers to search for.

        sort: boolean
            if ture, sort the index result by size, starting from the smallest.

        Returns
        -------
        Numpy array with index positions of the searched wavenumbers.
        """
        ret = find_nearest_index(self.wavenumber, wavenumbers)
        if sort:
            ret = np.sort(ret)
        return ret

    def wavenumber2pixelSlice(self, sl):
        """Get pixels from wavenumber slice"""
        return slice(*self.wavenumbers2index([sl.stop, sl.start]))

    @property
    def roi_x_wavenumber_spec(self):
        sl = self.roi_x_pixel_spec
        x = self.wavenumber[sl]
        return slice(int(x.min()), int(x.max()))

    @roi_x_wavenumber_spec.setter
    def roi_x_wavenumber_spec(self, value):
        self.roi_x_pixel_spec = slice(*self.wavenumbers2index([value.stop, value.start]))

    @property
    def rois_x_wavenumber_trace(self):
        """x rois in wavenumber coordinates."""
        ret = []
        for sl in self.rois_x_pixel_trace:
            ret.append(
                slice(int(self.wavenumber[sl].min()),
                      int(self.wavenumber[sl].max()))
            )
        return ret

    @rois_x_wavenumber_trace.setter
    def rois_x_wavenumber_trace(self, value):
        self.rois_x_pixel_trace = []
        for sl in value:
            self.rois_x_pixel_trace.append(
                slice(*self.wavenumbers2index([sl.stop, sl.start]))
            )

    @property
    def pumped_index(self):
        """Index value of the pumped data set.

        The y_pixel/spectra index value of the pumped data set.
        Upon importing the data this must be set manually, or 0
        is assumed.
        """
        return self._pumped_index

    @pumped_index.setter
    def pumped_index(self, value):
        if not value <= self.number_of_y_pixel:
            raise IOError("Cant set pumped index bigger then data dim.")
        self._pumped_index = value

    @property
    def unpumped_index(self):
        """y_pixel/spectra index of the unpumped data.

        Must be set during data import or a default of 1 is used."""
        return self._unpumped_index

    @unpumped_index.setter
    def unpumped_index(self, value):
        if not value <= self.number_of_spectra:
            raise IOError("Cant set unpumped index bigger then data dim.")
        self._unpumped_index = value

    def pumped(self, prop='normalized', **kwgs):
        """Returns subselected_data at pumped index.

        kwgs are same as for SfgRecord.select.
        overwritten defaults are:
        *prop*: normalized
        *frame_med*: True
        """
        kwgs.setdefault('roi_spectra', [self.pumped_index])
        kwgs['prop'] = prop
        # Reset subselect kwgs, to cope with multiple callings of it.
        kwgs.setdefault('roi_delay', slice(None))
        kwgs.setdefault('roi_frames', slice(None))
        kwgs.setdefault('roi_pixel', slice(None))
        return self.select(**kwgs)

    def unpumped(self, prop='normalized', **kwgs):
        kwgs.setdefault('roi_spectra', [self.unpumped_index])
        kwgs['prop'] = prop
        # Reset subselect kwgs, to cope with multiple callings of it.
        kwgs.setdefault('roi_delay', slice(None))
        kwgs.setdefault('roi_frames', slice(None))
        kwgs.setdefault('roi_pixel', slice(None))
        return self.select(**kwgs)

    def bleach(self, opt='rel', prop='normalized', **kwgs):
        """Calculate bleach of property with given operation."""

        kwgs['prop'] = prop
        # Reset subselect kwgs, to cope with multiple callings of it.
        kwgs.setdefault('roi_delay', slice(None))
        kwgs.setdefault('roi_frames', slice(None))
        kwgs.setdefault('roi_pixel', slice(None))
        pumped = self.pumped(**kwgs)
        unpumped = self.unpumped(**kwgs)

        if "relative" in opt or '/' in opt or 'rel' in opt:
            relative = True
            bleach = pumped / unpumped
        elif "absolute" in opt or '-' in opt or 'abs' in opt:
            relative = False
            bleach = pumped - unpumped
        else:
            raise IOError(
                "Must enter valid opt {} is invalid".format(opt)
            )

        if self.zero_time_subtraction:
            zero_time = bleach[self.zero_time_selec].mean(0)
            bleach -= zero_time
            # Recorretion for zero_time offset needed because
            # data is expected to be at 1 for negative times.
            if relative:
                bleach += 1
                self.zero_time_rel = zero_time
            else:
                # TODO add the mean of zero_time
                self.zero_time_abs = zero_time

        # Correct infs and nans.
        np.nan_to_num(bleach, copy=False)
        return bleach

    def contour(
            self,
            y_property='wavenumber',
            z_property='bleach',
            **subselect_kws
    ):
        """Returns data formatted for a contour plot.


        subselect_kws get passed to SfgRecord.select.
        defaults are adjusted with:
        *y_property*: bleach_rel
        *medfilt_kernel*: 5
        *resample_freqs*: 30
        Further susbelect_kws are:
          *roi_delay*:  roi of delays
          *roi_frames*:
          *roi_spectra*:
          *roi_pixel*:
          *frame_med*:
          *delay_mean*:
          *spectra_mean*:
          *pixel_mean*:
        """
        subselect_kws.setdefault('prop', z_property)
        subselect_kws.setdefault('medfilt_pixel', 11)
        subselect_kws.setdefault('frame_med', True)

        x = self.select('pp_delays', roi_delay=subselect_kws.get('roi_delay'))
        y = self.select(y_property, roi_pixel=subselect_kws.get('roi_pixel'))
        z = self.select(**subselect_kws)
        z = z.squeeze().T
        if len(z.shape) !=2:
            raise IOError(
                "Shape of subselected data can't be processed. Subselection was {}".format(subselect_kws)
            )
        return x, y, z

    def trace(
            self,
            prop='bleach',
            prop_kwgs={'opt': 'rel', 'prop': 'basesubed'},
            roi_wavenumber=None,
            roi_delay=None,
            roi_spectra=None,
            frame_med=True,
            shift_neg_time=False,
            roi_pixel=None,

    ):
        """Shortcut to get traces.

        prop: property to calculate the trace from
        prop_
        roi_wavenumber: roi to calculate trace over in wavenumbers.
        roi_delay: pp_delay roi to use as x axis.
        frame_med: calculate frame median.
        shift_neg_time: The Zero_time_subtraction can lead to an not 1 negative time.
            This corrects the while data set in such a way, that the given number of
            shift neg time points is used to move the complete data set such that it
            is around 1 there.
        roi_pixel: roi in pixel coordinates. Defaults to SfgRecord.roi_x_pixel_spec if
            neither roi_pixel nor roi_wavenumber is given.
        """
        if roi_wavenumber:
            roi_pixel = self.wavenumber2pixelSlice(roi_wavenumber)
        else:
            if not roi_pixel:
                roi_pixel = self.roi_x_pixel_spec
        x = self.select(prop='pp_delays', roi_delay=roi_delay)
        print(prop, prop_kwgs, roi_pixel)
        y = self.select(
            prop=prop,
            prop_kwgs=prop_kwgs,
            roi_pixel=roi_pixel,
            roi_delay=roi_delay,
            pixel_mean=True,
            frame_med=frame_med,
            roi_spectra=roi_spectra,
        )
        print(y.max())
        if shift_neg_time:
            y += 1 - y[:shift_neg_time].mean()
        yerr = self.sem(
            prop=prop,
            prop_kwgs=prop_kwgs,
            roi_pixel=roi_pixel,
            roi_delay=roi_delay,
            pixel_mean=True, roi_spectra=roi_spectra)
        return x, y, yerr

    def get_linear_baseline(self, start_slice=None,
                            stop_slice=None, data_attr="rawData"):
        """Calculate a linear baseline from data.

        Not Implemented yet."""
        data = getattr(self, data_attr)

        if isinstance(start_slice, type(None)):
            start_slice = slice(0, 0.1*self.pixel)

        if isinstance(stop_slice, type(None)):
            stop_slice = slice(-0.1*self.pixel, None)

        yp = np.array(
            [np.median(data[:, :, :, start_slice], -1),
             np.median(data[:, :, :, stop_slice], -1)]
        )
        xp = [0, self.pixel]
        x = np.arange(self.pixel)
        # TODO Make use of gridspec or any other multitimensional
        # linear interpolation method.
        raise NotImplementedError

    @property
    def static_corr(self):
        """Result after static correction."""
        if isinstance(self._static_corr, type(None)):
            self._static_corr = self.get_static_corr()
        return self._static_corr

    def get_static_corr(self):
        """Correction factors assuming area per pump sfg is constant.

        Assuming the area per pump sfg is constant, one can deduce correction
        faktors. That cope with the drifting of the Laser and height to some
        extend.

        Returns
        -------
        Correction factors in the same shape as SfgRecord.rawData
        """
        if not self.isBaselineSubed:
            msg = "Baseline not Subtracted. Cant calculate static sfg"\
                  "correction factors. Returning unity."
            warnings.warn(msg)
            return np.ones_like(self.rawData)
        # Medfilt makes algorithm robust agains Spikes.
        medfilt_kernel = (1, 1, 9)
        data = medfilt(self.unpumped, medfilt_kernel)
        area = data.sum(X_PIXEL_INDEX)
        # Correction factor is deviation from the mean value
        correction_factors = area / area.mean()
        correction_factors = correction_factors.reshape((
            self.number_of_pp_delays, self.number_of_frames, 1, 1
        )) * np.ones_like(self.rawData)
        return correction_factors

    def _readData(self):
        """The central readData function.

        This function shows the structure of a general data import.
        First one needs to find out what type the data is of.
        Then the data is imported and at the end metadata is extracted.

        Think of this function as a general recipe how to import data.
        It is here for structural reasons."""
        self._get_type()
        self._import_data()
        self._read_metadata()

    def _get_type(self):
        """Get the type of the data, by looking at its name, and if
        necessary by looking at a fraction of the data itself.

        We don't directly import all data here, so we can keep the
        actual import functions within there own modules and have
        more encapsulation. By reading only a fraction of the data
        we make sure the function is fast."""
        fhead, ftail = path.split(self._fname)

        # spe is binary and we hope its not named wrongly
        if path.splitext(ftail)[1] == '.spe':
            self._type = 'spe'
            return True

        # Compressed binary version.
        if path.splitext(ftail)[1] == '.npz':
            self._type = 'npz'
            return True

        # We open the file and by looking at the
        # first view lines, we can see if that is a readable file
        # and what function is needed to read it.
        start_of_data = np.genfromtxt(self._fname, max_rows=3, dtype="long")
        if start_of_data.shape[1] == 4:
            self._type = 'victor'
            return True

        elif start_of_data.shape[1] == 6:
            self._type = 'veronica'
            return True
        else:
            # Check if we have a header.
            # Only data from victor_controller has # started header.
            with open(self._fname) as f:
                line = f.readline()
                if line[0] == "#":
                    # First line is pixel then 3 spectra repeating
                    if (start_of_data.shape[1] - 1) % 3 == 0:
                        self._type = 'victor'
                        return True
                else:
                    if start_of_data.shape[1] % 6 == 0:
                        self._type = 'veronica'
                        return True
        raise IOError("Cant understand data in %f" % self._fname)

    def _import_data(self):
        """Import the data."""

        # Pick import function according to data type automatically.
        if self._type == "spe":
            from .io.spe import PrincetonSPEFile3
            self._sp = PrincetonSPEFile3(self._fname)
            self.rawData = self._sp.data.reshape(
                1, self._sp.NumFrames, self._sp.ydim, self._sp.xdim
            )
            return

        if self._type == "npz":
            imp = np.load(self._fname)
            for key, value in self.saveable.items():
                if key in imp.keys():
                    setattr(self, value, imp[key])
                    this = getattr(self, value)
                    rshape = getattr(this, 'shape')
                    if rshape == ():
                        setattr(self, value, this[()])
            try:
                self._rawDataE = imp['rawDataE']
                self._normE = imp['normE']
                self._basesubedE = imp['basesubedE']
                self._normalizedE = imp['normalizedE']
                self._baseline_offset = imp['baseline_offset']
            except KeyError:
                pass
            return

        if self._type == "veronica":
            self.rawData, self.pp_delays = get_from_veronika(self._fname)
            return

        if self._type == "victor":
            self.rawData, self.pp_delays = get_from_victor_controller(
                self._fname
            )
            return

        msg = "Uuuups this should never be reached."\
              "Bug with %s. I cannot understand the datatype" % self._fname

        raise NotImplementedError(msg)

    def _read_metadata(self):
        """Read metadata of the file"""

        from .utils.metadata import get_metadata_from_filename

        # Update datadependent rois
        if isinstance(self.roi_spectra, type(slice(None))):
           if not self.roi_spectra.stop:
               self.roi_spectra = slice(self.roi_spectra.start,
                                        self.number_of_spectra)
        if not self.roi_frames.stop:
            self.roi_frames = slice(self.roi_frames.start,
                                    self.number_of_frames)
        if not self.roi_delay.stop:
            self.roi_delay = slice(self.roi_delay.start,
                                   self.number_of_pp_delays)

        if self._type == "npz":
            # We skipp this step here, bacuse metadata is extracted from the
            # file Directly.
            return

        try:
            metadata = get_metadata_from_filename(self._fname)
        # TODO Refactor this, if I would program better this
        # would not happen
        except ValueError:
            msg ='ValueError while trying to extract metadata from filepath.'\
                '/nSkipping'
            warnings.warn(msg)

        if self._type == "victor":
            # Read metadata from file header.
            from .io.victor_controller import (read_header,
                                               translate_header_to_metadata)
            header = read_header(self._fname)
            metadata = {**metadata, **translate_header_to_metadata(header)}

        if self._type == 'spe':
            metadata['central_wl'] = self._sp.central_wl
            metadata['exposure_time'] = self._sp.exposureTime
            metadata['gain'] = self._sp.gain
            metadata['sp_type'] = 'spe'
            metadata['date'] = self._sp.date
            metadata['tempSet'] = self._sp.tempSet
            self._wavelength = self._sp.wavelength
            self.calib_poly = self._sp.calib_poly
            # Dont need the spe datatype object any more.
            del self._sp

        for key in metadata:
            self.metadata[key] = metadata[key]

        # Explicitly given VIS_WL overwrite file vis_wl.
        if not isinstance(VIS_WL, type(None)):
            self.metadata["vis_wl"] = VIS_WL
        # Explicitly given pump_freq overwrites file pump_freq
        if not isinstance(PUMP_FREQ, type(None)):
            self.metadata["pump_freq"] = PUMP_FREQ

    def copy(self):
        """Make a copy of the SfgRecord obj.

        This ensures, that we can have multiple disjoint objects, that
        come from the same file. By copying the data from the RAM, we can
        save some IO and speed up the process."""
        ret = SfgRecord()
        ret.pp_delays = self.pp_delays.copy()
        ret.metadata = self.metadata.copy()
        ret._fname = self._fname
        ret._type = self._type
        ret._unpumped_index = self._unpumped_index
        ret._pumped_index = self._pumped_index
        ret.baseline_offset = self.baseline_offset
        ret.zero_time_selec = self.zero_time_selec
        ret.zero_time_subtraction = self.zero_time_subtraction
        ret.baseline_offset = self.baseline_offset
        ret.roi_x_pixel_spec = self.roi_x_pixel_spec
        ret.rois_x_pixel_trace = self.rois_x_pixel_trace
        ret.roi_spectra = self.roi_spectra
        ret.roi_frames = self.roi_frames
        ret.roi_delay = self.roi_delay
        ret.rois_delays_pump_probe = self.rois_delays_pump_probe
        return ret

    def save(self, file, *args, **kwargs):
        """Save the SfgRecord into a compressed numpy array.

        Saves the `SfgRecord` obj into a compressed numpy array,
        that can later be reloaded and that can be used for further
        analysis. It is in particluar usefull to save data together
        with most of its properties like normalization and background
        spectra and also to save averaged results.

        If you want to know what is saved, then you can open the saved
        result with e.g. 7.zip and inspect its content."""
        kwgs = {key: getattr(self, value) for key, value in self.saveable.items()}
        print(kwgs['zero_time_selec'])
        np.savez_compressed(
            file,
            **kwgs
        )

    def keep_frames(self, frame_slice=None):
        """Resize data such, that only frame slice is leftover.

        frame_slice: Slice of frames to keep.
            if not given self.roi_frames is used.

        """
        if not frame_slice:
            frame_slice = self.roi_frames
        ret = self.copy()
        ret.rawData = self.rawData[:, frame_slice]
        ret.base = self.base[:, frame_slice]
        ret.norm = self.norm[:, frame_slice]
        ret.roi_frames = slice(None)

        return ret

    def make_avg(self, correct_static=True):
        """Returns an frame wise averaged SfgRecord.

        correct_static: boolean
           Toggle to use the area of the unpumped SFG signal
           as a correction factor to all spectra recorded at the same time.
        """
        ret = SfgRecord()
        ret.metadata = self.metadata
        ret.dates = [self.dates[0]]
        ret._unpumped_index = self._unpumped_index
        ret._pumped_index = self._pumped_index
        ret.pp_delays = self.pp_delays
        for key in ("rawData", "norm", "base"):
            setattr(
                ret, key, np.expand_dims(np.median(getattr(self, key), 1), 1)
            )
        for key in ('rawDataE', 'normE', 'baseE',
                    'basesubedE', 'normalizedE'):
            setattr(ret, key, getattr(self, key))
        if correct_static:
            pass
        return ret

    def make_static_corr(self):
        """Return a SfgRecord, that is corrected with the Static SFG assumption.

        The static SFG Assumption is, that in a Pump-Probe Experiment, the area
        of the static SFG Spectrum should be constant. A correction factor is
        calculated based on this assumption and applied on the data.

        This works only correctly if the baseline is set properly, and the
        unpumped index is set correctly.

        Returns
        -------
            SfgRecord with corrected data.
        """

        ret = SfgRecord()
        ret.rawData = self.rawData.copy()
        ret.metadata = self.metadata
        ret.dates = self.dates
        ret._unpumped_index = self._unpumped_index
        ret._pumped_index = self._pumped_index
        ret.pp_delays = self.pp_delays
        ret.base = self.base
        ret.norm = self.norm
        ret.normE = self.normE
        ret.basesubedE = self.basesubedE
        ret.rawDataE = self.rawDataE
        ret.normalizedE = self.normalizedE

        # Manipulate rawData
        delta = self.basesubed - (self.basesubed * self.static_corr)
        ret._rawData += delta
        # Reset internal properties so we leave with a clean SfgRecord
        return ret


def concatenate_list_of_SfgRecords(list_of_records):
    """Concatenate SfgRecords into one big SfgRecord."""

    concatable_attributes = ('rawData', 'base', 'norm')

    ret = SfgRecord()
    ret.metadata["central_wl"] = None
    # TODO Rewrite this pythonic
    for attribute in concatable_attributes:
        setattr(
            ret,
            attribute,
            np.concatenate(
                [getattr(elm, attribute) for elm in list_of_records],
                FRAME_AXIS_INDEX
            )
        )
    ret.dates = np.concatenate([elm.dates for elm in list_of_records]).tolist()
    ret.pp_delays = list_of_records[0].pp_delays
    #ret.metadata["central_wl"] = list_of_records[0].metadata.get("central_wl")
    #ret.metadata["vis_wl"] = list_of_records[0].metadata.get("vis_wl")
    #all([record.metadata.get(key) for record in list_of_records for key in list_of_records.metadata])
    ## Keep unchanged metadata and listify changed metadata.
    for key in list_of_records[0].metadata:
        values = [record.metadata.get(key) for record in list_of_records]
        if all([elm == values[0] for elm in values]):
            ret.metadata[key] = values[0]
        else:
            ret.metadata[key] = values
    return ret

def SfgRecords_from_file_list(list):
    """Import a list of files as a single SfgRecord.

    list: list of filepaths to import SfgRecords from.
    """
    return concatenate_list_of_SfgRecords([SfgRecord(elm) for elm in list])
