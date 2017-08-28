from os import path
import warnings

import numpy as np
from scipy.signal import medfilt
from scipy.stats import sem
import matplotlib.pyplot as plt

from .io.veronica import pixel_to_nm, get_from_veronika
from .io.victor_controller import get_from_victor_controller
from .utils import (
    nm_to_ir_wavenumbers, nm_to_wavenumbers, X_PIXEL_INDEX, Y_PIXEL_INDEX,
    FRAME_AXIS_INDEX, PIXEL, PP_INDEX, SPEC_INDEX, find_nearest_index
)
from .utils.consts import (VIS_WL, PUMP_FREQ, NORM_SPEC, BASE_SPEC,
                           FRAME_AXIS_INDEX_P,  PP_INDEX_P)


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

        # 1d array with wavenumber values
        self._wavenumber = None

        # Error of absolute bleach
        self._bleach_absE = None

        # Error of realtive bleach
        self._bleach_relE = None

        # Error of normalized bleach
        self._bleach_abs_normE = None

        # 3d Array of pumped data
        self._pumped = None

        # 3d Array of unpumped data
        self._unpumped = None

        # 3d array of pumped and normalized
        self._pumped_norm = None

        # 3d array of unpumped and normalized
        self._unpumped_norm = None

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

        # Region of interest x_pixel
        self.rois_x_pixel_trace = [slice(None, None)]

        # Region of interest for the x_spec
        self.roi_x_pixel_spec = slice(None, None)

        # Region of interest y_pixel/spectra
        self.roi_spectra = slice(None, None)

        # Region of interest frames
        self.roi_frames = slice(None, None)

        # Region of interest pp_delays
        self.roi_delays = slice(None, None)

        # Subreions of interest for pump probe
        self.rois_delays_pump_probe = [slice(None, None)]

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
        if not isinstance(base, type(None)):
            if isinstance(base, str):
                base = SfgRecord(base).data
                self.base = base
            else:
                self.base = base

        if isinstance(norm, type(None)):
            norm = NORM_SPEC
        if not isinstance(norm, type(None)):
            if isinstance(norm, str):
                norm = SfgRecord(norm).data
                self.norm = norm
            else:
                self.norm = norm

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
        self._pumped = None
        self._unpumped = None
        self._pumped_norm = None
        self._unpumped_norm = None
        self.isBaselineSubed = False
        self.isNormalized = False

    @property
    def rawDataE(self):
        """Std error of the mean rawData.

        See `SfgRecord.rawData` for further information."""
        if isinstance(self._rawDataE, type(None)):
            self._rawDataE = np.divide(
                self.rawData.std(FRAME_AXIS_INDEX),
                np.sqrt(self.number_of_frames)
            )
            self._rawDataE = np.expand_dims(self._rawDataE, 1)
        return self._rawDataE

    @rawDataE.setter
    def rawDataE(self, value):
        """See `SfgRecord.rawData` for further information."""
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._rawDataE = value

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
        self._pumped = None
        self._unpumped = None
        self._pumped_norm = None
        self._unpumped_norm = None

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
        self._pumped = None
        self._unpumped = None
        self._pumped_norm = None
        self._unpumped_norm = None

    @property
    def baseE(self):
        """Error of the baseline/background.

        4d array like `SfgRecord.rawData` wit error of base"""
        if isinstance(self._baseE, type(None)):
            self._baseE = np.divide(
                self.base.std(FRAME_AXIS_INDEX),
                np.sqrt(self.base.shape[FRAME_AXIS_INDEX])
            )
            self._baseE = np.expand_dims(self._baseE, 1)
        return self._baseE

    @baseE.setter
    def baseE(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._baseE = value

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
    def normE(self):
        """Error of the normalization spectrum.

        Same structure as `SfgRecord.rawData`."""
        if isinstance(self._normE, type(None)):
            self._normE = np.divide(
                self.norm.std(FRAME_AXIS_INDEX),
                np.sqrt(self.norm.shape[FRAME_AXIS_INDEX])
            )
            self._normE = np.expand_dims(self._normE, 1)
        return self._normE

    @normE.setter
    def normE(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._normE = value

    @property
    def data(self):
        """Buffer for processed data.

        A buffer to store processed data in. You can put rawData, or
        baseline subtract data or normalized data in this buffer. It
        is mostly a convenience thing during interactive sessions.

        Same structure asself._norm `SfgRecord.rawData`."""
        ret = self._data
        if isinstance(ret, type(None)):
            ret = self.rawData
        return ret

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._data = value

    @property
    def dates(self):
        """List of datetimes the spectra were recorded at.

        A list of datetime objects when each spectrum was created."""
        if isinstance(self._dates, type(None)):
            self._dates = np.arange(
                    (self.number_of_pp_delays * self.number_of_frames)
                )
            self._dates = self.metadata["date"] + self._dates *\
                self.metadata["exposure_time"]
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def times(self):
        """List of timedeltas the spectra were recorded with."""
        ret = []
        time_of_a_scan = self.metadata['exposure_time']
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[0]):
                ret.append(
                    (i * self.number_of_pp_delays + j) * time_of_a_scan
                )
        return ret

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
        return self.data.shape[FRAME_AXIS_INDEX]

    @property
    def pixel(self):
        """Iterable list of pixels."""
        return np.arange(self.data.shape[X_PIXEL_INDEX])

    @property
    def number_of_y_pixel(self):
        """Number of y_pixels/spectra."""
        return self.data.shape[Y_PIXEL_INDEX]

    @property
    def number_of_spectra(self):
        """Number of spectra/y_pixels."""
        return self.number_of_y_pixel

    @property
    def number_of_x_pixel(self):
        """Number of pixels."""
        return self.data.shape[X_PIXEL_INDEX]

    @property
    def number_of_pp_delays(self):
        """Number of pump probe time delays."""
        return self.data.shape[PP_INDEX]

    @property
    def zero_time_subtraction(self):
        return self._zero_time_subtraction

    @zero_time_subtraction.setter
    def zero_time_subtraction(self, value):
        self._zero_time_subtraction = value
        # Reset dependen properties
        self._pumped = None
        self._unpumped = None
        self._pumped_norm = None
        self._unpumped_norm = None

    @property
    def zero_time_selec(self):
        """Slice/List to select delays for zero time subtraction."""
        return self._zero_time_selec

    @zero_time_selec.setter
    def zero_time_selec(self, value):
        self._zero_time_selec = value
        self._pumped = None
        self._unpumped = None
        self._pumped_norm = None
        self._unpumped_norm = None

    @property
    def central_wl(self):
        """Central wavelength of the grating in nm."""
        return self.metadata.get("central_wl")

    @central_wl.setter
    def central_wl(self, value):
        self.metadata["central_wl"] = value

        # For spe files we must rely on the stored
        # wavelength because  we cannot recalculate,
        # since calibration parameters are not allways
        # known. From spe version 3 on they are amongst
        # the metadata, but that is not taken into account
        # here.
        if self.metadata.get("sp_type"):
            return
        self._wavelength = None
        self._wavenumber = None

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
        if isinstance(self._wavelength, type(None)):
            cw = self.metadata["central_wl"]
            self._wavelength = self.get_wavelength(cw)
        return self._wavelength

    @wavelength.setter
    def wavelength(self, arg):
        self._wavelength = arg

    def get_wavelength(self, cw):
        """Get wavelength in nm.

        Parameters:
        -----------
        cw: number
            central wavelength of the camera.
        """
        ret = self.pixel.copy()
        if isinstance(cw, type(None)) or cw < 1:
            return ret
        ret = pixel_to_nm(self.pixel, cw)
        return ret

    @property
    def vis_wl(self):
        """Wavelength of the visible in nm."""
        return self.metadata.get("vis_wl")

    @vis_wl.setter
    def vis_wl(self, value):
        self.metadata['vis_wl'] = value
        # Must reset wavenumber
        self._wavenumber = None

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

        if isinstance(self._wavenumber, type(None)):
            vis_wl = self.metadata.get("vis_wl")
            self._wavenumber = self.get_wavenumber(vis_wl)
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, arg):
        """Setter for wavenumber propertie."""

        self._wavenumber = arg

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

    def get_wavenumber(self, vis_wl):
        """return calculated wavenumbers."""

        # Reversed, because if vis_wl or central_wl are useless,
        # we have at least flipped the spectrum.
        #ret = self.pixel[::-1]
        if isinstance(vis_wl, type(None)) or vis_wl < 1:
            ret = nm_to_wavenumbers(self.wavelength)
        else:
            ret = nm_to_ir_wavenumbers(self.wavelength, vis_wl)
        return ret

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

    def _spectra_selector(self, propery=''):
        data = getattr(self, propery)
        return data[
            self.roi_delays,
            self.roi_frames,
            self.roi_spectra,
            self.rois_x_pixel[0]
        ]

    @property
    def basesubed(self):
        """Baselinesubstracted data.

        The rawData after baseline subtraction."""

        if isinstance(self._basesubed, type(None)):
            self._basesubed = self.get_baselinesubed()
        return self._basesubed

    @property
    def basesubedE(self):
        """Data error after subtracting baseline."""
        if isinstance(self._basesubedE, type(None)):
            self._basesubedE = np.sqrt(self.rawDataE**2 + self.baseE**2)
        return self._basesubedE

    @basesubedE.setter
    def basesubedE(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._basesubedE = value

    def get_baselinesubed(self, use_rawData=True):
        """Get baseline subtracted data."""

        if use_rawData or self.isNormalized:
            ret = self.rawData - self.base
        else:
            ret = self.data - self.base
        return ret

    def sub_base(self, inplace=False, use_rawData=False):
        """subsitute baseline of data

        Use SfgRecord.base and substitute it from SfgRecord.data.

        Parameters
        ----------
        inplace: boolean
            If true, subsitute SfgRecord.base from SfgRecord.data inplace.

        use_rawData: boolean
            Subtract baseline from `SfgRecord.rawData` and not
            `SfgRecord.data`.
            If data is already normalized (`SfgRecord.isNormalized` is True)
            then, this is always calculated from `SfgRecord.rawData`. So be
            carefull about this when using filters, because they usually only
            work on `SfgRecord.data` and not `SfgRecord.rawData`.

        Returns
        -------
        Numpy array with SfgRecod.data subsituted by SfgRecord.base"""
        ret = self.get_baselinesubed(use_rawData)
        if inplace and not self.isBaselineSubed:
            self.data = ret
            # Toggle prevents multiple baseline substitutions
            self.isBaselineSubed = True
        return ret

    def add_base(self, inplace=False):
        """Add baseline to data.

        Can be used to readd the baseline after subtraction the baseline."""
        ret = self.data + self.base
        if inplace and not self.isBaselineSubed:
            self.data = ret
            self.isBaselineSubed = False
        return ret

    @property
    def normalized(self):
        """Normalized data.

        4D array like `SfgRecord.rawData` after normalizting.

        Examples
        --------
        >>> import sfg2d
        >>> bg = sfg2d.SfgRecod('path/to/baselinedata').make_avg()
        >>> q0 = sfg2d.SfgRecod('path/to/quartz').make_avg()
        >>> q0.base = bg.rawData
        >>> data = sfg2d.SfgRecod('/path/to/data')
        >>> data.norm = q0.basesubed
        >>> data.base = bg.rawData
        >>> data.normalized
        """
        if isinstance(self._normalized, type(None)):
            self._normalized = self.get_normalized()
            number_of_nan_infs = np.count_nonzero(
                np.isnan(self._normalized) | np.isinf(self._normalized)
               )
            self.normalized_nan_infs = number_of_nan_infs
            np.nan_to_num(self._normalized, copy=False)

        return self._normalized

    def get_normalized(self, use_rawData=True):
        """Get normalized spectrum.
        Parameters
        ----------
        use_rawData: boolean
            Use `SfgRecod.rawData` to calculate result
            else use `SfgRecod.data`
        Return
        ------
        4d-numpy array with normalized data.
        """

        if use_rawData:
            ret = (self.rawData - self.base) / self.norm
        else:
            ret = self.data / self.norm

        return ret

    def normalize(self, inplace=False, use_rawData=True):
        """Normalize data.

        Parameters
        ----------
        inplace : boolean default False
            apply restult to `SfgRecod.data` buffer.
        use_rawData : boolean default True
            Recalculate normalized data using `SfgRecod.rawData`.
        """
        if not self.isBaselineSubed:
            warnings.warn(
                "Normalizing data although baseline is not substracted."
                "Consider subtracting baseline with"
                "SfgRecod.sub_base(inplace=True) first."
            )
        ret = self.get_normalized(use_rawData)
        if inplace and not self.isNormalized:
            self.data = ret
            self.isNormalized = True
        return ret

    def un_normalize(self, inplace=False):
        """un normalize `SfgRecod.data`

        Reverse the normalization applied to `SfgRecod.data`.

        Parameters
        ----------
        inplace: default False boolean
            Apply to `SfgRecod.data`

        Returns
        -------
        The unnormalized version of the data.
        """
        if not self.isBaselineSubed:
            msg = "Normalizing data although baseline is not substracted."\
                  "Consider subtracting baseline with"\
                  "`SfgRecod.sub_base(inplace=True)` first."
            warnings.warn(msg)
        ret = self.data * self.norm
        if inplace and self.isNormalized:
            self.data = ret
            self.isNormalized = False
        return ret

    @property
    def normalizedE(self):
        """Error of the normalized data."""
        if isinstance(self._normalizedE, type(None)):
            norm = np.expand_dims(self.norm.mean(FRAME_AXIS_INDEX), 1)
            basesubed = np.expand_dims(self.basesubed.mean(FRAME_AXIS_INDEX), 1)
            self._normalizedE = np.sqrt(
                (self.basesubedE/norm)**2 + ((basesubed/(norm)**2 * self.normE))**2
            )
        return self._normalizedE

    @normalizedE.setter
    def normalizedE(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._normalizedE = value

    @property
    def sum_argmax(self, frame_median=True,
                   pixel_slice=slice(None, None)):
        """argmax of pp_delay with biggest sum.

        frame_median : bool
            default true, Take median of frames.
            if false give a framewise argmax."""
        if frame_median:
            ret = np.median(
                self.data[:, :, :, pixel_slice], 1
            ).sum(-1).argmax(0)
        else:
            ret = self.data[:, :, :, pixel_slice].sum(-1).argmax(0)
        return ret

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
        # Because we set a new index bleach and pumped must be recalculated.
        self._pumped = None
        self._pumped_norm = None

    @property
    def pumped(self):
        """Pumped data.

        3d Array of the pumped data. If no pumped data is set explicit,
        `SfgRecord.pumped_index` is used to get the pumped data."""
        if isinstance(self._pumped, type(None)):
            self.pumped = self.pumped_index
        return self._pumped

    @pumped.setter
    def pumped(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self.pumped_index = value
            # by using slice we keep data 4dim
            sl = slice(value, value + 1)
            self._pumped = self.basesubed[:, :, sl]
        else:
            raise NotImplemented()

    @property
    def pumped_norm(self):
        """Pumped data normalized."""
        if isinstance(self._pumped_norm, type(None)):
            sl = slice(self.pumped_index, self.pumped_index+1)
            self._pumped_norm = self.normalized[:, :, sl]
        return self._pumped_norm

    @pumped_norm.setter
    def pumped_norm(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self.pumped_index = value
            sl = slice(value, value + 1)
            self._pumped_norm = self.normalized[:, :, sl]
        else:
            raise NotImplemented()

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
        # Beacause we setted a new index on the unpumped spectrum we must
        # reset the bleach.
        self._unpumped = None
        self._unpumped_norm = None

    @property
    def unpumped(self):
        """Unpumped data array.

        3d numpy array with the unpumed data.
        Uses `SfgRecord.unpumped_index`, to obtain unpumped data.
        """
        if isinstance(self._unpumped, type(None)):
            self.unpumped = self.unpumped_index
        return self._unpumped

    @unpumped.setter
    def unpumped(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self.unpumped_index = value
            sl = slice(value, value + 1)
            self._unpumped = self.basesubed[:, :, sl]
        else:
            raise NotImplemented()

    @property
    def unpumped_norm(self):
        """Unpumped data normalized."""
        if isinstance(self._unpumped_norm, type(None)):
            self.unpumped_norm = self.unpumped_index
        return self._unpumped_norm

    @unpumped_norm.setter
    def unpumped_norm(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self.unpumped_index = value
            sl = slice(value, value + 1)
            self._unpumped_norm = self.normalized[:, :, sl]
        else:
            raise NotImplemented()

    def _calc_bleach(
            self,
            operation,
            pumped=None,
            unpumped=None,
            normalized=False,
            zero_time_subtraction=True,
            nan2num=True,
    ):
        """Calculate bleach using the given operation.

        operation: string
            possible are: 'absolute', 'realtive', '-' or '/'
        pumped: None, index or array
            If None SfgRecord.pump is used. Else SfgRecord.pump is setted
            by pumped
        unpumped: None, index or array
            Same as pumped only for unpumped
        normalized:
            use normalized data to calculate bleach.
        zero_time_subtraction : boolean
            substitute the first spectrum from the data to account for
            the constant offset.
        nan2num: Boolean
            If True replace nans with 0 and infs with finite numbers.
            Only used during calculation of realtive bleach.

        Returns
        -------
        The calculated bleach result.
        """
        bleach = np.zeros((
            self.number_of_pp_delays,
            self.number_of_frames,
            1,
            self.number_of_x_pixel,
        ))

        if self.number_of_spectra < 2:
            warnings.warn("Not enough spectra to calculate bleach.")
            return bleach

        if "relative" in operation or '/' in operation:
            relative = True
        elif "absolute" in operation or '-' in operation:
            relative = False
        else:
            raise IOError(
                "Must enter valid operation {} is invalid".format(operation)
            )

        # Reset pumped and umpumped properties
        # to ensure latest data is used.
        # pumped and umpumped are index numbers
        if not isinstance(pumped, type(None)):
            self.pumped = pumped
            self.pumped_norm = pumped
        if not isinstance(unpumped, type(None)):
            self.unpumped = unpumped
            self.unpumped_norm = unpumped

        if normalized:
            pumped = self.pumped_norm
            unpumped = self.unpumped_norm
        else:
            pumped = self.pumped
            unpumped = self.unpumped

        if relative:
            # Prevent division by zero
            # Because our measurment is not that good. We can safely do this.
            unpumped[np.isin(unpumped, 0)] = 1e-08
            bleach = pumped / unpumped
        else:
            bleach = pumped - unpumped

        if zero_time_subtraction:
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
        return bleach

    def calc_bleach_abs(self, pumped=None, unpumped=None,
                        normalized=False, zero_time_subtraction=True):
        """Calculate absolute bleach.

        pumped: None, index or array
            If None SfgRecod.pump is used. Else SfgRecord.pump is setted
            by pumped
        unpumped: None, index or array
            Same as pumped only for unpumped
        zero_time_subtraction : boolean
            substitute the first spectrum from the data to account for
            the constant offset.

        Returns
        -------
        The calculated 3d bleach result.
        """
        return self._calc_bleach('absolute', pumped, unpumped,
                                 normalized, zero_time_subtraction)

    def calc_bleach_rel(self, pumped=None, unpumped=None,
                        normalized=False, zero_time_subtraction=True,):
        """Calculate the relative bleach

        Parameters
        ----------
        pumped: index or None
            y_pixel/spectra index of the pumped spectrum. Default is None,
            and will make it use `SfgRecord._pumped_index`
        unpumped: index or None
            like pumped only for unpumped None defaults to
            `SfgRecord._unpumped_index`
        normalized: boolean
            Use normalized data.
        zero_time_subtraction: bollean default true
            subtract the 0th pp_delay index. This corrects for constant
            offset between pumped and unpumped data.
        """
        return self._calc_bleach(
            'relative',
            pumped,
            unpumped,
            normalized,
            zero_time_subtraction=zero_time_subtraction
        )

    @property
    def bleach_abs(self):
        """Absolute bleached data.

        3d array with bleached data result.
        The Absolute bleach is calculated by subtracting two
        baselinesubstracted pumped and unpumped signal.

        """
        ret = self.calc_bleach_abs(
            normalized=False,
            zero_time_subtraction=self.zero_time_subtraction
        )
        return ret

    @property
    def bleach_abs_norm(self):
        ret = self._bleach_abs_norm = self.calc_bleach_abs(
            normalized=True,
            zero_time_subtraction=self.zero_time_subtraction
        )
        return ret

    @property
    def bleach_rel(self):
        """Relative bleached data

        3d array with relative bleached data.

        """
        ret = self.calc_bleach_rel(
            normalized=False,
            zero_time_subtraction=self.zero_time_subtraction
        )
        return ret

    @property
    def bleach_rel_norm(self):
        """Relative bleach from normalized data."""
        ret = self.calc_bleach_rel(
            normalized=True,
            zero_time_subtraction=self.zero_time_subtraction
        )
        return ret

    # TODO add a frame median method and use keepdims in trace calculation.
    def trace(
            self,
            attr="bleach_rel",
            delay_slice=slice(None, None),
            frame_slice=slice(None, None),
            spectra_slice=slice(None, None),
            x_pixel_slice=slice(None, None),
    ):
        """Return traces of the SfgRecord.

        A trace is an mean over a pixel region and an median of a frame
        region.

        attr: string
            attribute to get the data from.
        x_pixel_slice: slice
            slice to calc mean over
        frame_slice: slice
            frame slice to calculate median over.
        spectra_slice: spectra of interest.

        Returns
        --------
        2d array with:
        [delays, specs]
        """
        data = getattr(self, attr)
        # Frame median must be calculated first, to get rid of spikes.
        # This is a violation of the programming idiom but its the only
        # way I could come up with a sensible soltuion.
        data = np.median(
            data[
                delay_slice,
                frame_slice,
                spectra_slice,
                x_pixel_slice
            ],
            FRAME_AXIS_INDEX_P,
        ).mean(X_PIXEL_INDEX)
        return data

    def _traces_property(self, attr):
        """Generator for the trace properties.
        attr: string
            attribute to get data of.

        Returns
        -------
        3d array with:
        [rois_x_pixel, delays, y_specs]
        """
        ret = []
        only_one_spec = ('bleach_', 'pumped_', 'unpumped_')
        if any([teststr in attr for teststr in only_one_spec]):
            roi_spectra = slice(None, None)
        else:
            roi_spectra = self.roi_spectra

        for roi_x_pixel in self.rois_x_pixel_trace:
            ret.append(
                self.trace(
                    attr=attr,
                    delay_slice=self.roi_delays,
                    frame_slice=self.roi_frames,
                    spectra_slice=roi_spectra,
                    x_pixel_slice=roi_x_pixel,
                )
            )
        return np.array(ret)

    @property
    def traces_bleach_abs(self):
        """trace of absolute bleach."""
        return self._traces_property("bleach_abs")

    @property
    def traces_bleach_abs_norm(self):
        """trace of normalized absolute bleach."""
        return self._traces_property("bleach_abs_norm")

    @property
    def traces_bleach_rel(self):
        """trace of relative bleach."""
        return self._traces_property("bleach_rel")

    @property
    def traces_bleach_rel_norm(self):
        """trace of relative bleach."""
        return self._traces_property("bleach_rel_norm")

    @property
    def traces_normalized(self):
        """trace of normalized data."""
        return self._traces_property("normalized")

    @property
    def traces_rawData(self):
        """trace of raw data."""
        return self._traces_property("rawData")

    @property
    def traces_basesubed(self):
        """trace of basesubed"""
        return self._traces_property("basesubed")

    def traceE(
            self,
            attr='bleach_rel',
            x_pixel_slice=slice(None, None),
            frame_slice=slice(None, None),
            spectra_slice=slice(None, None),
            delay_slice=slice(None, None),
    ):
        """Std error of the mean for a trace.

        Currently this is not an exact result, because normalization and
        background subtraction are ignored.

        Calculated from frames variations.

        Returns
        --------
        2d array with [delays, spectras]
        """
        data = getattr(self, attr)
        # I dont think this is correct but for now okay.
        data = np.mean(data[delay_slice, :, spectra_slice, x_pixel_slice],
                       X_PIXEL_INDEX)
        data = sem(data[:, frame_slice], FRAME_AXIS_INDEX_P)
        return data

    def _tracesE_property(self, attr):
        """Generator for the Trace property.

        It makes use of the same reshaping as SfgRecord._trace_property.

        Returns
        --------
        3d array with [rois_x_pixel, delays, spectras]
        """
        ret = []
        for roi_x_pixel in self.rois_x_pixel:
            ret.append(
                self.traceE(
                    attr,
                    roi_x_pixel,
                    self.roi_frames,
                    self.roi_spectra,
                    self.roi_delays,
                )
            )
        return np.array(ret)

    @property
    def tracesE_bleach_abs(self):
        return self._tracesE_property(attr="bleach_abs")

    @property
    def tracesE_bleach_abs_norm(self):
        return self._tracesE_property(attr="bleach_abs_norm")

    @property
    def tracesE_bleach_rel(self):
        return self._tracesE_property(attr="bleach_rel")

    @property
    def tracesE_bleach_rel_norm(self):
        return self._tracesE_property(attr="bleach_rel_norm")

    @property
    def tracesE_normalized(self):
        return self._tracesE_property(attr="normalized")

    @property
    def tracesE_rawData(self):
        return self._tracesE_property(attr="rawData")

    @property
    def tracesE_basesubed(self):
        return self._tracesE_property(attr="basesubed")

    def _time_track(self, property="basesubed", roi_x_pixel=slice(None, None)):
        """Return a time track.

        A time track is the mean of each spectrum vs its temporal position
        index. In other words, it tells you when and for how long the
        Measurement was stable."""
        data = getattr(self, property)
        ret = data[:, :, :, roi_x_pixel].mean(-1).reshape(
                (self.number_of_pp_delays*self.number_of_frames,
                 self.number_of_spectra),
                order="F"
        )
        return ret

    def _frame_track(
            self,
            property='basesubed',
            roi_x_pixel=slice(None, None),
            roi_delays=slice(None, None),
    ):
        """A frame track.

        Frame track is an pixel wise and delay wise average of the data."""
        data = getattr(self, property)
        data = data[roi_delays, :, :, roi_x_pixel].mean(PP_INDEX_P).mean(X_PIXEL_INDEX)
        return data

    @property
    def frame_track_basesubed(self):
        return self._frame_track(
            property="basesubed",
            roi_x_pixel=self.roi_x_pixel_spec,
            roi_delays=self.roi_delays
        )[self.roi_frames, self.roi_spectra]

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
            self.rawData = imp['rawdata']
            self.wavenumber = imp['wavenumber']
            self.wavelength = imp['wavelength']
            self.base = imp['base']
            self.metadata = imp['metadata'][()]
            self.dates = imp['dates']
            self.norm = imp['norm']
            self.pp_delays = imp['pp_delays']
            self._unpumped_index = imp['_unpumped_index'][()]
            self._pumped_index = imp['_pumped_index'][()]
            # The following only try because I have data files
            # that were created without these properties and I
            # don't want to break them.
            try:
                self.rawDataE = imp['rawDataE']
                self.normE = imp['normE']
                self.basesubedE = imp['basesubedE']
                self.normalizedE = imp['normalizedE']
                self.baseline_offset = imp['baseline_offset']
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
        ret._data = self._data.copy()
        ret._type = self._type
        ret._dates = self._dates
        ret._wavelength = self.wavelength.copy()
        ret._wavenumber = self.wavenumber.copy()
        ret._unpumped_index = self._unpumped_index
        ret._pumped_index = self._pumped_index
        ret.baseline_offset = self.baseline_offset
        return ret

    def save(self, file, *args, **kwargs):
        """Save the SfgRecord into a compressed numpy array.

        Saves the `SfgRecord` obj into a compressed numpy array,
        that can later be reloaded and that can be used for further
        analysis. It is in particluar usefull to save data together
        with mist of its properties like normalization and background
        spectra and also to save averaged results.

        If you want to know what is saved, then you can open the saved
        result with e.g. 7.zip and inspect its content."""
        np.savez_compressed(
            file,
            rawdata=self.rawData,
            norm=self.norm,
            base=self.base,
            pp_delays=self.pp_delays,
            wavelength=self.wavelength,
            wavenumber=self.wavenumber,
            metadata=self.metadata,
            dates=self.dates,
            _pumped_index=self._pumped_index,
            _unpumped_index=self._unpumped_index,
            rawDataE=self.rawDataE,
            normE=self.normE,
            baseE=self.baseE,
            basesubedE=self.basesubedE,
            normalizedE=self.normalizedE,
            baseline_offset=self.baseline_offset,
        )

    def keep_frames(self, frame_slice=slice(None)):
        """Resize data such, that only frame slice is leftover."""
        ret = self.copy()
        ret.rawData = self.rawData[:, frame_slice]
        ret.base = self.base[:, frame_slice]
        ret.norm = self.norm[:, frame_slice]

        return ret

    def make_avg(self, correct_static=True):
        """Returns an frame wise averaged SfgRecord.

        correct_static: boolean
           Toggle to use the area of the unpumped SFG signal
           as a correction factor to all spectra recorded at the same time.
        """
        ret = SfgRecord()
        ret.metadata = self.metadata
        ret.wavelength = self.wavelength
        ret.wavenumber = self.wavenumber
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
        ret.wavelength = self.wavelength
        ret.wavenumber = self.wavenumber
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

    def subselect(
            self,
            y_property="basesubed",
            x_property='pixel',
            roi_delays=None,
            roi_frames=None,
            roi_spectra=None,
            roi_x_pixel_spec=None,
            frame_med=True,
            delay_mean=False,
            spectra_mean=False,
            pixel_mean=False,
            medfilt_pixel=1,
    ):
        """Select subset of data and apply comom operations.

        Subselection of data is done by using rois.
        x_rois_elms subselectrs the SfgRecord.rois_x_pixel property for
        this plot.

        Parameters
        ----------
        y_property: y attribute to subselect.
        roi_delays: roi of delays to take into account
        roi_frames: roi of frames to take into account
        roi_spectra: roi of spectra to take into account
        roi_x_pixel_spec: roi of x_pixel to take into account
        frame_med: Calculate median over roi_frames
        delay_mean: Calculate median over delay_roi
        spectra_mean: Calculate mean over spectra
        medfilt_pixel: Number of pixel moving median filter applies to.
            Must be an uneven number.

        Returns tuple of arrays
        1d array with selected x_data,
        4d numpy array. [pp_delay, frames, spectra, pixel]
        with y_data

        """

        if isinstance(roi_delays, type(None)):
            roi_delays = self.roi_delays
        if isinstance(roi_frames, type(None)):
            roi_frames = self.roi_frames
        if isinstance(roi_spectra, type(None)):
            only_one_spec = ('bleach', 'pumped', 'unpumped')
            if any([teststr in y_property for teststr in only_one_spec]):
                roi_spectra = [0]
            else:
                roi_spectra = self.roi_spectra
        if isinstance(roi_x_pixel_spec, type(None)):
            roi_x_pixel_spec = self.roi_x_pixel_spec

        x_data = getattr(self, x_property)[roi_x_pixel_spec]
        y_data = getattr(self, y_property)[
            roi_delays,
            roi_frames,
            roi_spectra,
            roi_x_pixel_spec,
        ]
        if frame_med:
            y_data = np.median(y_data, FRAME_AXIS_INDEX, keepdims=True)
        if delay_mean:
            y_data = np.mean(y_data, PP_INDEX, keepdims=True)
        if spectra_mean:
            y_data = np.mean(y_data, SPEC_INDEX, keepdims=True)
        if medfilt_pixel != 1:
            y_data = medfilt(y_data, (1, 1, 1, medfilt_pixel))
        if pixel_mean:
            y_data = np.mean(y_data, X_PIXEL_INDEX, keepdims=True)
        return x_data, y_data

    def plot_spec(
            self,
            y_property="basesubed",
            x_property='pixel',
            roi_delays=None,
            roi_frames=None,
            roi_spectra=None,
            roi_x_pixel_spec=None,
            frame_med=True,
            delay_mean=False,
            spectra_mean=False,
            pixel_mean=False,
            medfilt_pixel=1,
            ax=None,
            **kwargs
    ):
        """Plot a spectra property.

        Subselection of data is done by using rois.
        x_rois_elms subselectrs the SfgRecord.rois_x_pixel property for
        this plot.

        Parameters
        ----------
        y_property: y attribute to plot
        x_property: x attribute to plot
        roi_delays: roi of delays to take into account
        roi_frames: roi of frames to take into account
        roi_spectra: roi of spectra to take into account
        roi_x_pixel_spec: roi of x_pixel to take into account
        frame_med: Calculate median over roi_frames
        delay_mean: Calculate median over delay_roi
        medfilt_kernel: Kernel of moving median filter moving over pixels.
            Must be an uneven number.
        **kwargs are passed to matplotlib plot.
        """
        if not ax:
            ax = plt.gca()

        x_data, y_data = self.subselect(
            y_property,
            x_property,
            roi_delays,
            roi_frames,
            roi_spectra,
            roi_x_pixel_spec,
            frame_med,
            delay_mean,
            spectra_mean,
            pixel_mean,
            medfilt_pixel,
            )
        for delay in y_data:
            for frame in delay:
                for spectrum in frame:
                    ax.plot(x_data, spectrum, **kwargs)

    def plot_bleach(
            self,
            y_property="bleach_abs",
            rois_delays=None,
            **kwargs
    ):
        """Plot Bleach of Record.

        Parameters
        ----------
        y_property: attribute to get data from
        rois_delays: List of slices or None.
            If None `SfgRecord.rois_delays_pump_probe`
            is used.
        **kwargs are passed to `SfgRecord.plot_spec`
        but roi_delays is overwritten with elements from
        `SfgRecord.rois_delays_pump_probe` and roi_spectra is
        forced to 1, because there is always only one spec
        """

        if not rois_delays:
            rois_delays = self.rois_delays_pump_probe
        for roi_delay in rois_delays:
            # There is always only one bleach spectrum.
            self.plot_spec(
                y_property=y_property,
                roi_delays=roi_delay,
                roi_spectra=[0],
                **kwargs,
               )

    def plot_trace(
            self,
            y_property="traces_rawData",
            frame_med=True,
            ax=None,
            **kwargs
    ):
        """Plot a trace"""

        if not ax:
            ax = plt.gca()

        x_data = self.pp_delays[self.roi_delays]
        # data is of shape [roi, pp_delay, spectra]
        y_data = getattr(self, y_property)

        for roi_index in range(len(y_data)):
            roi_slice = self.rois_x_pixel_trace[roi_index]
            start, stop = [self.wavenumber[roi_slice][elm] for elm in (0, -1)]
            label = '{:.0f}-{:.0f} 1/cm'.format(start, stop)
            ax.plot(
                x_data,
                y_data[roi_index], "-o",
                label=label,
                **kwargs
            )

    def plot_time_track(
            self,
            y_property='basesubed',
            x_property='index',
            roi_x_pixel=slice(None, None),
            roi_spectra=slice(None, None),
            ax=None,
            **kwargs
    ):
        """Make a plot of the time track data."""
        if not ax:
            ax = plt.gca()

        data = self._time_track(
            property=y_property, roi_x_pixel=roi_x_pixel
        )[:, roi_spectra]
        ax.plot(data, **kwargs)


def concatenate_list_of_SfgRecords(list_of_records):
    """Concatenate SfgRecords into one big SfgRecord."""

    concatable_attributes = ('rawData', 'base', 'norm')

    ret = SfgRecord()
    ret.metadata["central_wl"] = None
    ret.wavelength = list_of_records[0].wavelength
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
