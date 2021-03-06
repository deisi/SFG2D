import warnings
import logging

import numpy as np
import scipy.fftpack as fft
from scipy.signal import medfilt
from scipy.stats import sem
import scipy.ndimage as nd
import dpath.util

from .io import import_data
from .utils import (
    nm_to_ir_wavenumbers, nm_to_wavenumbers,  find_nearest_index, pixel_to_nm,
)
from sfg2d.utils.config import CONFIG
from .utils.filter import double_resample

VIS_WL = CONFIG['VIS_WL']
PUMP_FREQ = CONFIG['PUMP_FREQ']
NORM_SPEC = CONFIG['NORM_SPEC']
BASE_SPEC = CONFIG['BASE_SPEC']
FRAME_AXIS_INDEX_P = CONFIG['FRAME_AXIS_INDEX_P']
PP_INDEX_P = CONFIG['PP_INDEX_P']
X_PIXEL_INDEX = CONFIG['X_PIXEL_INDEX']
Y_PIXEL_INDEX = CONFIG['Y_PIXEL_INDEX']
FRAME_AXIS_INDEX = CONFIG['FRAME_AXIS_INDEX']
PIXEL = CONFIG['PIXEL']
PP_INDEX = CONFIG['PP_INDEX']
SPEC_INDEX = CONFIG['SPEC_INDEX']
logger = logging.getLogger(__name__)


def import_sfgrecord(
        record,
        baseline=None,
        norm=None,
        kwargs_select_baseline=None,
        kwargs_select_norm=None,
        **kwargs
):
    """Function to import and configure SfgRecord.

    **Arguments:**
      - **record**:
         If string given, the file gets loaded as record.
         If list of strings given, these files get loaded
         and concatenated as record.

    **Keywords:**
      - **baseline**: String or SfgRecord to use as baseline
      - **norm**: String or SfgRecord to use for normalization
      - **kwargs_select_baseline**: Keywords to subselect baseline record with.
          Default is {'prop': 'rawData', 'frame_med': True}
      - **norm_select_kw**: Keywords to subselect norm record with.
          Default is {'prop': 'basesubed', 'frame_med': True}

    **kwargs**:
      Get passes as attributes to the record.

    **Return:**
      SfgRecord obj

    """

    if not kwargs_select_baseline:
        kwargs_select_baseline = {}
    if not kwargs_select_norm:
        kwargs_select_norm = {}
    if isinstance(record, str):
        record = SfgRecord(record)
    elif hasattr(record, '__iter__'):
        records = []
        for elm in record:
            records.append(SfgRecord(elm))
        record = concatenate_list_of_SfgRecords(records)
    elif isinstance(record, type(SfgRecord)):
        pass
    else:
        raise NotImplementedError(
            'Type of record {} not implemented'.format(record))

    if isinstance(baseline, str):
        baseline = SfgRecord(baseline)

    if baseline:
        kwargs_select_baseline.setdefault('prop', 'rawData')
        kwargs_select_baseline.setdefault('frame_med', True)
        record.base = baseline.select(**kwargs_select_baseline)

    if isinstance(norm, str):
        norm = SfgRecord(norm)
        try:
            norm.base = baseline.select(**kwargs_select_baseline)
            logger.debug('Warning: Using record baseline as normalization baseline.')
        except AttributeError:
            logger.warn('Warning: No baseline for normalization found.')

    if norm:
        kwargs_select_norm.setdefault('prop', 'basesubed')
        kwargs_select_norm.setdefault('frame_med', True)
        record.norm = norm.select(**kwargs_select_norm)

    if kwargs:
        for key, value in kwargs.items():
            setattr(record, key, value)

    return record

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

    **Properties**
      - **fname**: String or list of strings.
          Path to input data
      - **raw_data**: 4 dim numpy array
          - axis 0: pump-probe time delay
          - axis 1: frames
          - axis 2: y-pixels
          - axis 3: x-pixels
      - **base**: 4d castable array or string to baseline file
          Data for the baseline
      - **norm**: 4d castable array or string with path to baseline file
          Data to normalize with
      - **baseline_offset**: Default 0
          Add static offset to baseline.
      - **wavelength**: np.array of wavelength data.
          If data is from an .spe file, wavelength can be read from it.
          If data is from a .dat file, wavelength gets calcuated based
          on a calibration line with the central wavelength taken from:
          `SfgRecord.metadata['central_wl']`
      - **wavenumber**: np.array
          Wavenumber calculated from the wavelength as described above.
          The effect of the up-conversion is calcula
          `SfgRecord.metadata["vis_wl"]` as wavelength of the visible.
      - **roi_x_pixel_spec**: slice
        region of interest for normal spectra. Usually given bey the grating
        and bandwidth of the setup plus central wavelength.
      - **norm_het_shift**: Shift heterodyne quartz by given angle in radiants.
      - **norm_het_start**: Starting frequency of heterodyne freq
      - **norm_het_stop**: Stop freq of heterodyne freq
      - **roi_frames**: Slice. Region of interes for frame axis
      - **zero_time_select**: default [0, 1]
          Array of indeces used for zero time selection.
      - **pump_freq**: Frequency of the pump
      - **pump_width**: Width of the pump
      - **name**: strign. Additional name of the sample like d2o or something.
      - **zero_time_subtraction**: boolean, default True
          If zero time should be subtracted.
      - **roi_delay**: slice. Region of interest for delays
      - **pumped_index**: index number of the pumped spectrum
      - **unpumped_index**: index of unpumped spectrum
      - **roi_spectra**: Region of interest for spectra axis.
      - **vis_wl**: Wavelength of visible light. Default is written from file
      - **replace_pixel**: list of tuples.
          ((index, value), (index1, value1), ...) to replace pixel at position           index with value, replacement happens through all axis.
      - **average_pixels**: Forgot what it was
      - **trim_frames**: slice or array of indeces.
          Is used after data import to trim down frame axis. Usefull when
          combining many different scans, with partial good and partial
          bad data.
      - **base_corrections**: list of dicts with
          ({"axis": axis_description, "value": value_to_set}). Usefull to cope
          with unstable baselines due to a bug in em-cm mode switching.
      - **kwargs**:
          get added to metadata
    """
    def __init__(self, fname=None, rawData=None,
                 base=None, norm=None, baseline_offset=0, wavelength=None,
                 wavenumber=None, roi_x_pixel_spec=None, het_shift=None,
                 het_start=None, het_stop=None, norm_het_shift=None,
                 norm_het_start=None, norm_het_stop=None, roi_frames=None,
                 zero_time_select=None, pump_freq=None, pump_width=None,
                 cc_width=None, cc_pos=None, name=None, zero_time_subtraction=None,
                 roi_delay=None, pumped_index=0, unpumped_index=1,
                 roi_spectra=None, vis_wl=None, replace_pixels=None,
                 average_pixels=None, trim_frames=None, base_corrections=None,
                 **kwargs,
    ):

        ## Beacaue I know it will happen and we cans safely deal with it.
        #np.seterr(divide='ignore')
        #np.seterr(invalid='ignore')

        # Set format and test filenames
        self.fname = fname

        # List of dates the spectra were recorded at.
        self._dates = None

        # PumpProbe delays
        self.pp_delays = np.array([0])

        # Error of raw data
        self._rawDataE = np.zeros((1, 1, 1, PIXEL))

        # 4d buffer array of processed data
        self._data = None

        # 4d array of baseline/background data
        self._base = None

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

        # Crosscorrelation of the data
        self.cc = None

        # boolean for normalization
        self.isNormalized = False

        # 4d array of normalized data
        self._normalized = None

        # error of normalized data
        self._normalizedE = None

        # type of data file
        self.type = 'unknown'

        # Dicitionary of metadata
        self.metadata = {}

        # Placeholderf for setted wavenumbers and wavelength
        self._setted_wavelength = None
        self._setted_wavenumber = None

        # y-pixel/spectra index of pumped data
        self._pumped_index = pumped_index

        # index of unpumped data
        self._unpumped_index = unpumped_index

        # Boolean to toggle default zero_time subtraction
        self._zero_time_subtraction = True

        # list/slice of delay indexes for zero_time_subtraction
        self._zero_time_select = [0, 1]

        # array of bleach value at negative time
        self.zero_time_abs = None

        # array of relative bleach value at negative time.
        self.zero_time_rel = None

        # array of normalized bleach at negative time.
        self.zero_time_norm = None

        # 4d array of values for the static drift correction.
        self._static_corr = None

        #roi_x_pixel_spec
        self.roi_x_pixel_spec = slice(0, PIXEL)

        # Region of interest x_pixel
        self.rois_x_pixel_trace = [slice(0, PIXEL)]

        # Region of interest y_pixel/spectra
        self.roi_spectra = slice(None)

        # Region of interest pp_delays
        self.roi_delay = slice(0, None)

        # Subreions of interest for pump probe
        self.rois_delays_pump_probe = [slice(0, None)]

        # Default roir frames
        self.roi_frames = slice(None)

        # Buffer for traces. kwg is binning and then [X, Y, Yerr]
        self.traces = {}

        # Buffer for fot models.
        self.models = {}

        # Buffer for figures
        self.figures = {}

        # A short name for the record
        if not name:
            self.name = ''
        else:
            self.name = name

        # A Nice Latex Version of the name of the recrod
        self._lname = None

        #
        self.het_shift = 0
        self.het_start = None
        self.het_stop = None

        self.norm_het_shift = 0
        self.norm_het_start = None
        self.norm_het_stop = None



        #### Import data from hdd
        if fname:
            self._readData(self.fname)
        #########################################################################
        # Here come properties that can overwrite the filename results after import

        # 4d array of raw data
        if hasattr(rawData, '__iter__'):
            self._rawData = rawData

        # Constant offset for baseline
        self._baseline_offset = baseline_offset

        # 1d array with wavelength values
        if hasattr(wavelength, '__iter__'):
            self.wavelength = wavelength

        # 1d array with wavenumber values
        if hasattr(wavenumber, '__iter__'):
            self.wavenumber = wavenumber

        # Trim frames away
        self.trim_frames(trim_frames)

        # Region of interest for the x_spec
        if roi_x_pixel_spec:
            self.roi_x_pixel_spec = roi_x_pixel_spec

        # Default time_domain start cutoff
        if het_start:
            self.het_start = het_start

        # Default time_domain stop cutoff
        if het_stop:
            self.het_stop = het_stop

        # Default shift of the heterodyne signal
        if het_shift:
            self.het_shift = het_shift

        # Default amount to shift heterodyne normalization with
        if norm_het_shift:
            self.norm_het_shift = norm_het_shift

        # Default time_domain start cutoff for heterodyne normalization signal
        if norm_het_start:
            self.norm_het_start = norm_het_start

        # Default time_domain stop cutoff for heterodyne normalization signal
        if norm_het_stop:
            self.norm_het_stop = norm_het_stop

        # Update zero time select
        if zero_time_select:
            self.zero_time_select = zero_time_select

        # Update zero time subtraction boolean
        if not isinstance(zero_time_subtraction, type(None)):
            self._zero_time_subtraction = zero_time_subtraction

        # Delete unneeded delays
        if roi_delay:
            self._rawData = self._rawData[roi_delay]
            #self.roi_delay = roi_delay

        # Import only interesting frames
        if roi_frames:
            self._rawData = self._rawData[:, roi_frames]

        # Delete unneeded spectra during import to reduce data amount
        if roi_spectra:
            self._rawData = self._rawData[:, :, roi_spectra]
            #self.roi_spectra = roi_spectra

        # vis_wl
        if vis_wl:
            self.vis_wl = vis_wl

        #Set pump_frequency
        if pump_freq:
            self.pump_freq = pump_freq

        # Set pump_width
        if pump_width:
            self.pump_width = pump_width

        if cc_width:
            self.cc_width = cc_width

        if cc_pos:
            self.cc_pos = cc_pos

        # Add all additional kwargs to metadata
        for item, value in kwargs.items():
            self.metadata[item] = value

        if isinstance(base, str):
            base = SfgRecord(base).rawData
            self.base = base
        elif not isinstance(base, type(None)):
            self.base = base

        if isinstance(norm, str):
            norm = SfgRecord(norm).basesubed
            self.norm = norm
        elif not isinstance(norm, type(None)):
            self.norm = norm

        if not isinstance(replace_pixels, type(None)):
            for pixel, value in replace_pixels:
                self._rawData[:, :, :, pixel] = value


        if not isinstance(average_pixels, type(None)):
            for pixel, region in average_pixels:
                self._rawData[:, :, :, pixel] = np.nanmedian(
                    self._rawData[:, :, :, pixel-region: pixel+region], -1
                       )

        # base correction to manipulte baseline offsets individually
        if not isinstance(base_corrections, type(None)):
            self.base_corrections = base_corrections
            for base_dict in base_corrections:
                axis = base_dict['axis']
                value = base_dict['value']
                self._base[axis] = self._base[axis] + value
            self._basesubed = None
            self._normalized = None

    def update_config(self, kwargs):
        """Update the configuration of the record with given kwargs"""
        logger.info('Updating config of record {}'.format(self))
        for key, value in kwargs.items():
            logger.debug('Updating {} with {}'.format(key,  value))
            setattr(self, key, value)

    @property
    def info(self):
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
        print(msg)

    @property
    def saveable(self):
        """A dict of the saveable properties.

        This dict translates between the propertie names and how they
        are saved in an npz."""

        # This is needed, because one might or might not circumvent the setter
        # and getter functions of the properties.
        return {
            'rawData': '_rawData',
            'metadata': 'metadata',
            'norm': '_norm',
            'base': '_base',
            'pp_delays': 'pp_delays',
            'pumped_index': '_pumped_index',
            'unpumped_index': '_unpumped_index',
            'baseline_offset': '_baseline_offset',
            'zero_time_subtraction': '_zero_time_subtraction',
            'zero_time_select': 'zero_time_select',
            'rois_x_pixel_trace': 'rois_x_pixel_trace',
            'roi_x_pixel_spec': 'roi_x_pixel_spec',
            'roi_spectra': 'roi_spectra',
            'roi_frames': 'roi_frames',
            'roi_delay': 'roi_delay',
            'rois_delays_pump_probe': 'rois_delays_pump_probe',
            'het_shift': 'het_shift',
            'het_start': 'het_start',
            'het_stop': 'het_stop',
            'norm_het_shift': 'norm_het_shift',
            'norm_het_start': 'norm_het_start',
            'norm_het_stop': 'norm_het_stop',
            'wavenumber': 'wavenumber',
            'wavelength': 'wavelength',
            'pump_freq': 'pump_freq',
        }

    @property
    def fname(self):
        """List of filenames to read data from."""
        return self._fname

    @fname.setter
    def fname(self, fname):
        self._fname = fname

    def trim_frames(self, index_list):
        if isinstance(index_list, type(None)):
            return
        logger.warn('Trimming data down with {}'.format(index_list))
        self._rawData = self._rawData[:, index_list, :, :]
        #self._base = self._base[:, index_list, :, :]
        #self._norm = self._norm[:, index_list, :, :]

    def select(
            self, prop="normalized", kwargs_prop=None,
            roi_delay=None, roi_frames=None, roi_spectra=None, roi_pixel=None,
            frame_med=False, delay_mean=False, spectra_mean=False,
            pixel_mean=False, medfilt_pixel=1, resample_freqs=0,
            attribute=None, scale=None, abs=False, square=False, sqrt=False,
            offset=None, roi_wavenumber=None, imag=False, real=False,
            sum_pixel=False, rebin=False, rebin_agg=np.mean, gaussian_filter=None,
            debug=False,
            **kwargs
    ):
        """Central Interface to select data.

        This is usually all you need.

        **Keywords:**
          - **prop**: Name of the propertie to select data from
          - **kwargs_prop**: For Infered properties, spectial kwargs,
              Unknown **kwargs get passed into this.
          - **roi_delay:** Region of Interest of delay slice
          - **roi_frames:** Select Frames
          - **roi_spectra:** Select Spectra
          - **roi_pixel:** Select Pixel
          - **frame_med:** Calculate frame wise median.
          - **delay_mean:** calculate delaywise median.
          - **spectr_mean:** Spctra wise mean
          - **pixel_mean:** Calculates median. Called mean due to historic reasons.
          - **medfilt_pixel:** Moving median filter over pixel axis.
          - **resample_freqs:** FFT Frequency filter. Give number of frequencoes.
              0 is off. Higher number -> Less smoothing.
          - **attribute**: Attribute to get at the end
          - **scale**: A factor to scale results with
          - **abs**: Boolean to get absolute with
          - **square**: Boolean to calculate square with
          - **sqrt**: Boolean to calculate square-root with
          - **offset**: Offset to add to result
          - **roi_wavenumber**: roi to calculate trace over in wavenumbers.
          - **imag**:  Boolean to get imaginary part
          - **real**: Boolean to get real part
          - **sum_pixel**: sum up pixels
          - **rebin**: Rebins pixel axis by given integer
          - **rebin_agg**: Defaults to np.mean, any other aggregating function
            would do the trick tough, e.g. np.median. It must have the
            signature (array, axis).

        **kwargs**:
          Combined into *kwargs_prop*

        **Returns:**
        4-D Array with [Delay, Frames, Spectra, Pixel] axes.
        """
        if not kwargs_prop:
            kwargs_prop = {}

        # For historic reasons kwargs_prop and kwargs are not the same
        kwargs_prop = {**kwargs, **kwargs_prop}
        logger.debug('select:kwargs_prop: {}'.format(kwargs_prop))

        # Internal definitions are used by default to make interface
        # as small simple as possible.
        if isinstance(roi_delay, type(None)):
            roi_delay = self.roi_delay
        if isinstance(roi_frames, type(None)):
            roi_frames = self.roi_frames
        if isinstance(roi_spectra, type(None)):
            roi_spectra = self.roi_spectra
        # For traces a different x_pixel region should be used.
        if isinstance(roi_pixel, type(None)):
            if roi_wavenumber:
                roi_pixel = self.wavenumber2pixelSlice(roi_wavenumber)
            else:
                roi_pixel = self.roi_x_pixel_spec

        # Usually X-Axis properties that dont need extra treatment
        if prop in ('pixel', 'wavenumber', 'wavelength', 'pp_delays', 'frames', 'pp_delays_ps', 'times'):
            ret = getattr(self, prop)
            if prop in ('pixel', 'wavenumber', 'wavelength'):
                ret = ret[roi_pixel]
            elif prop in ("pp_delays", 'pp_delays_ps'):
                ret = ret[roi_delay]
            elif prop is "frames":
                ret = ret[roi_frames]
            if rebin:
                logger.debug('Rebinning pixels with {}'.format(int(rebin)))
                logger.debug('Rebinning pixel with {} as aggregation function'.format(rebin_agg))
                ret = rebin_agg(ret.reshape(
                        (ret.shape[-1]//int(rebin), int(rebin))
                      ), -1)
            return ret

        elif prop == 'range':
            ret = np.arange(0, self.number_of_x_pixel)
            if roi_pixel:
                ret = np.arange(0, roi_pixel.stop - roi_pixel.start)

            if rebin:
                logger.debug('Rebinning range with {}'.format(int(rebin)))
                logger.debug('Rebinning range with {} as aggregation function'.format(rebin_agg))
                ret = rebin_agg(ret.reshape(
                        (ret.shape[-1]//int(rebin), int(rebin))
                      ), -1)
            return ret

        # Real properties get used directly
        elif prop in ('rawData', 'basesubed', 'normalized', 'base', 'norm',
                      'chi2'):
            ret = getattr(self, prop)
            # Only Real properties get cut down
            logger.debug('Prop: {}'.format(prop))
            logger.debug('Selecting Delay {}'.format(roi_delay))
            logger.debug('Selecting Frames: {}'.format(roi_frames))
            logger.debug('Selecting Spectra: {}'.format(roi_spectra))
            logger.debug('Selecting Pixels: {}'.format(roi_pixel))
            ret = ret[
                    roi_delay,
                    roi_frames,
                    roi_spectra,
                    roi_pixel,
                   ]
            logger.debug('Shape after slicing: {}'.format(ret.shape))

        # Infered properites need aditional kwargs
        elif prop in ('pumped', 'unpumped', 'bleach', 'trace',
                      'time_domain', 'frequency_domain', 'normalize_het',
                      'norm_het', 'signal_het', 'track'):
            # rois get passed down to inferior properties
            kwargs_prop['roi_delay'] = roi_delay
            kwargs_prop['roi_frames'] = roi_frames
            kwargs_prop['roi_spectra'] = roi_spectra
            kwargs_prop['roi_pixel'] = roi_pixel

            # Corrects trace in such a way, that it returns only the y data.
            # This is the expected behavior of a select call.
            if prop == 'trace':
                kwargs_prop.setdefault('y_only', True)

            ret = getattr(self, prop)(**kwargs_prop)

            # Pixel get selected up on trace selection already. Here they must
            # be removed, because else a double call would occur
            #if prop == 'trace':
            # Sublevel props dont get used for selection
        else:
            raise ValueError("Don't know prop: {}".format(prop))

        logger.debug('Prop: {}'.format( prop ))
        logger.debug('ret.shape: {}'.format( ret.shape ))
        if prop in ('normalize_het', 'norm_het', 'signal_het') and frame_med:
            ret = np.nanmean(ret, axis=FRAME_AXIS_INDEX, keepdims=True)
            frame_med = None
        if frame_med:
            logger.debug('Calculating frame_mean')
            ret = np.nanmedian(ret, FRAME_AXIS_INDEX, keepdims=True)
        if delay_mean:
            logger.debug('Calculating delay_mean')
            ret = np.nanmean(ret, PP_INDEX, keepdims=True)
        if spectra_mean:
            logger.debug('Calculating spectra_mean')
            ret = np.nanmean(ret, SPEC_INDEX, keepdims=True)
        # Median because sometimes there are still nan and infs left.
        if pixel_mean:
            logger.debug('Calculating pixel_mean')
            ret = np.nanmedian(ret, X_PIXEL_INDEX, keepdims=True)
        if medfilt_pixel > 1:
            logger.debug('Calculating medfilt_pixel: {}'.format(medfilt_pixel))
            ret = medfilt(ret, (1, 1, 1, medfilt_pixel))
        if resample_freqs:
            logger.debug('Calculating resample_freqs {}'.format(
                resample_freqs)
            )
            ret = double_resample(ret, resample_freqs, X_PIXEL_INDEX)
        if gaussian_filter:
            logger.debug('Filtering with gaussian filter {}'.format(
                gaussian_filter)
            )
            ret = nd.gaussian_filter(ret, gaussian_filter)
        if scale:
            logger.debug('Sacaling with: {}'.format(scale))
            ret *= scale
        if offset:
            if np.any(np.iscomplex(ret)):
                offset = np.complex(offset, offset)
            logger.debug('Using offset: {}'.format( offset ))
            ret += offset
        if attribute:
            logger.debug("Attribute {}".format(attribute))
            ret = getattr(ret, attribute)
        if abs:
            logger.debug("Calculating abs")
            ret = np.absolute(ret)
        if imag:
            logger.debug("Calculating imag")
            ret = np.imag(ret)
        if real:
            logger.debug("Calculating real")
            ret = np.real(ret)
        if square:
            logger.debug("Calculating square")
            ret = np.square(ret)
        if sqrt:
            logger.debug("Calculating sqrt")
            ret = np.sqrt(ret)
        if sum_pixel:
            logger.debug("Calculating sum_pixel")
            ret = np.sum(ret, axis=X_PIXEL_INDEX, keepdims=True)
        if rebin:
            # Rebins over pixel axis with given rebin number
            logger.debug('Rebinning Data with {}'.format(int(rebin)))
            logger.debug('Rebinning Data with {} as aggregation function'.format(rebin_agg))
            ret = rebin_agg(ret.reshape(
                (*ret.shape[:-1], ret.shape[-1]//int(rebin), int(rebin))
            ), -1)

        return ret

    def sem(self, prop, **kwargs):
        """Returns standard error of the mean of given property.

        kwargs: get passed to SfgRecord.subselect
        """
        #Forced because error is calculated over the frame axis.
        kwargs['frame_med'] = False
        kwargs['prop'] = prop
        if 'trace' in prop:
            kwargs['pixel_mean'] = True
        return sem(self.select(**kwargs), FRAME_AXIS_INDEX)


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
        Uses SfgRecord.basesubed and SfgRecord.norm to normalize data.

        4D array like `SfgRecord.rawData` after normalizing.
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
        self._rawDataE = self.sem('rawData')
        return self._rawDataE

    @property
    def baseE(self):
        """Error of the baseline/background.
        """
        self._baseE = self.sem('base')
        return self._baseE

    @property
    def normE(self):
        """Error of the normalization spectrum.

        Same structure as `SfgRecord.rawData`."""
        self._normE = self.sem('norm')
        return self._normE

    @property
    def basesubedE(self):
        """Data error after subtracting baseline."""
        self._basesubedE = self.sem('basesubed')
        return self._basesubedE

    @property
    def normalizedE(self):
        """Error of the normalized data."""
        self._normalizedE = self.sem('normalized')
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

    def pixels2wavenumbers(self, pixels):
        """Convert a given list of pixels to wavenumbers.

        **Arguments:**
          - **pixels**: List of pixels.

        **Returns:**
          List of wavenumber values.
        """

        cw = self.central_wl
        wavelength = pixel_to_nm(pixels, cw)
        vis_wl = self.metadata.get("vis_wl")
        if isinstance(vis_wl, type(None)) or vis_wl < 1:
            wavenumber = nm_to_wavenumbers(wavelength)
        else:
            wavenumber = nm_to_ir_wavenumbers(wavelength, vis_wl)
        return wavenumber

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
        if self.type == 'spe':
            return self._sp.wavelength
        cw = self.central_wl
        self._wavelength = pixel_to_nm(self.pixel, cw)
        return self._wavelength

    @wavelength.setter
    def wavelength(self, arg):
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
        logger.debug('Setting pump_freq to: {}'.format(value))
        self.metadata['pump_freq'] = value

    @property
    def cc_width(self):
        return self.metadata.get('cc_width')

    @cc_width.setter
    def cc_width(self, value):
        logger.debug('Setting cc_witch to: {}'.format(value))
        self.metadata['cc_width'] = value

    @property
    def cc_pos(self):
        return self.metadata.get('cc_pos')

    @cc_pos.setter
    def cc_pos(self, value):
        logger.debug('Setting cc_pos to: {}'.format(value))
        self.metadata['cc_pos'] = value

    @property
    def pump_width(self):
        return self.metadata.get('pump_width')

    @pump_width.setter
    def pump_width(self, value):
        logger.debug('Setting pump_width to: {}'.format(value))
        self.metadata['pump_width'] = value

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
        # Changing this because I dont know if this will still be needed
        # vis_wl = self.metadata.get("vis_wl")
        vis_wl = self.vis_wl
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
    def zero_time_select(self):
        """Slice/List to select delays for zero time subtraction."""
        return self._zero_time_select

    @zero_time_select.setter
    def zero_time_select(self, value):
        self._zero_time_select = value

    @property
    def lname(self):
        lname_dict = {
            'd2o': 'D$_2$O',
            'na2so4': 'Na$_2$SO$_4$',
            'na2co3': 'Na$_2$CO$_3$',
        }
        if not self._lname:
            for test_str in lname_dict.keys():
                if test_str in self.name:
                    return lname_dict[test_str]
            return ''
        return self._lname

    @lname.setter
    def lname(self, value):
        self._lname = value

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

    def pumped(self, prop='normalized', pumped_index=None, **kwargs):
        """Returns subselected_data at pumped index.

        kwargs are same as for SfgRecord.select.
        overwritten defaults are:
        *prop*: normalized
        *pumped_index*: Index of pumped spectrum. Default to
            SfgRecord.pumped_index

        **kwargs**: get passes to SfgRecord.select.
        """
        if not pumped_index:
            pumped_index = self.pumped_index
        kwargs['roi_spectra'] = slice(pumped_index, pumped_index + 1)
        kwargs['prop'] = prop
        return self.select(**kwargs)

    def unpumped(self, prop='normalized', unpumped_index=None, **kwargs):
        """See SfgRecord.pumped."""
        if not unpumped_index:
            unpumped_index = self.unpumped_index
        kwargs['roi_spectra'] = slice(unpumped_index, unpumped_index + 1)
        kwargs['prop'] = prop
        return self.select(**kwargs)

    def bleach(self, opt='rel', kwargs_pumped=None, kwargs_unpumped=None, **kwargs):
        """Calculate bleach of property with given operation.

        **Keywords:**
          - **opt**: Operation to calculate bleach with. Valid are 'rel' or
            'abs'. For relative bleach aka division of pumped and unpumped.
            absolute bleach is the difference of pumped - unpumped.

        **kwargs** get passed to select call of pumped and unpumped selection.

        **Returns**:
        4 D array in the usual from but spec index length will always be 1.
        """

        # Must be applied after zero_time_subtraction
        # Othwerwise zero_time_subtraction fails if
        # roi_delay is provided
        try:
            roi_delay = kwargs.pop('roi_delay')
        except KeyError:
            roi_delay = self.roi_delay

        if not kwargs_pumped:
            kwargs_pumped = {}
        kwargs_pumped = {**kwargs, **kwargs_pumped}
        kwargs_pumped['prop'] = 'pumped'
        if not kwargs_unpumped:
            kwargs_unpumped = {}
        kwargs_unpumped = {**kwargs, **kwargs_unpumped}
        kwargs_unpumped['prop'] = 'unpumped'

        if opt == 'rel':
            # Use basesubed instead of norm. This does not update, but only creates new
            dpath.util.new(kwargs_pumped, 'kwargs_prop/prop', 'basesubed')
            dpath.util.new(kwargs_unpumped, 'kwargs_prop/prop', 'basesubed')

        pumped = self.select(**kwargs_pumped)
        unpumped = self.select(**kwargs_unpumped)

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
            zero_time = bleach[self.zero_time_select].mean(0)
            bleach -= zero_time
            # Recorretion for zero_time offset needed because
            # data is expected to be at 1 for negative times.
            if relative:
                bleach += 1
                self.zero_time_rel = zero_time
            else:
                # TODO add the mean of zero_time
                self.zero_time_abs = zero_time

        bleach = bleach[roi_delay]
        # Correct infs and nans.
        np.nan_to_num(bleach, copy=False)
        return bleach

    def contour(
            self,
            kwargs_xdata=None,
            kwargs_ydata=None,
            kwargs_zdata=None,
    ):
        """Returns data formatted for a contour plot.

        kwargs get passed to SfgRecord.select.
        defaults are adjusted with:
        *y_property*: bleach_rel
        *medfilt_kernel*: 11
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
        if not kwargs_xdata:
            kwargs_xdata = {}
        if not kwargs_ydata:
            kwargs_ydata = {}
        if not kwargs_zdata:
            kwargs_zdata = {}

        # Warn if unused parameters get used
        if kwargs_ydata.get('roi_frames'):
            Warning('roi frames in ydata gets overwritten by roi_frames \
            from zdata')
        if kwargs_zdata.get('roi_pixel'):
            Warning('roi_pixel of z data gets overwritten by roi_pixel of \
            ydata')
        if kwargs_zdata.get('roi_delay'):
            Warning('roi_delay of z data gets overwritten by roi_delay of \
            xdata')

        # Must translate wavenumbers to pixel for y at this time,
        # So it autoamtically gets synced with zdata in the parameter
        # setting step
        roi_pixel = kwargs_ydata.get('roi_pixel')
        roi_wavenumber = kwargs_ydata.get('roi_wavenumber')
        if isinstance(roi_pixel, type(None)):
            if roi_wavenumber:
                roi_pixel = self.wavenumber2pixelSlice(roi_wavenumber)
                kwargs_ydata['roi_pixel'] = roi_pixel

        # These parameters must match. Else cant produce a contour
        # plot.
        kwargs_ydata['roi_frames'] = kwargs_zdata.get('roi_frames')
        kwargs_zdata['roi_pixel'] = kwargs_ydata.get('roi_pixel')
        kwargs_zdata['roi_delay'] = kwargs_xdata.get('roi_delay')

        kwargs_xdata.setdefault('prop', 'pp_delays')
        kwargs_ydata.setdefault('prop', 'wavenumber')
        kwargs_ydata.setdefault('frame_med', True)
        kwargs_zdata.setdefault('prop', 'bleach')
        kwargs_zdata.setdefault('frame_med', True)
        kwargs_zdata.setdefault('medfilt_pixel', 7)
        kwargs_zdata.setdefault('resample_ferq', 15)

        x = self.select(**kwargs_xdata)
        y = self.select(**kwargs_ydata)
        z = self.select(**kwargs_zdata)
        return x, y, z

    def trace(
            self,
            prop='bleach',
            kwargs_prop=None,
            roi_delay=None,
            shift_neg_time=False,
            y_only=False,
            yerr_only=False,
            squeeze=False,
            **kwargs
    ):
        """Shortcut to get trace.
        Trace means, that the mean over a certain pixel region is calculated.
        E.g. this function calls select and allways add the `pixel_mean`
        option.

        prop: property to calculate the trace from
        kwargs_prop: additional kwargs of the property
        roi_delay: pp_delay roi to use as x axis.
        shift_neg_time: Zero_time_subtraction can lead to negative time not beeing 1.
            This corrects the data set in such a way, that the given number of
            shift neg time points is used to move the complete data set such that it
            is around 1 there.
        y_only: lets this function only return the y data.
        yerr_only: return only yerror
        squeeze: Allows to squeeze the y and yerr result
        **kwargs get passed to `SfRecord.select()`:

        """
        if not kwargs_prop:
            kwargs_prop = {}
        kwargs_prop.setdefault('opt', 'rel')
        kwargs_prop.setdefault('prop', 'basesubed')

        kwargs['pixel_mean'] = True
        kwargs['roi_delay'] = roi_delay

        logger.debug('trace:{}'.format(kwargs_prop))
        y = self.select(
            prop=prop,
            kwargs_prop=kwargs_prop,
            **kwargs
        )
        x = self.select(prop='pp_delays', roi_delay=roi_delay)

        if shift_neg_time:
            y += 1 - y[:shift_neg_time].mean()

        kwargs['frame_med'] = False
        yerr = self.sem(
            prop=prop,
            kwargs_prop=kwargs_prop,
            **kwargs
        )
        if squeeze:
            y = y.squeeze()
            yerr = yerr.squeeze()
        if y_only:
            return y
        if yerr_only:
            return yerr
        return x, y, yerr

    def traceE(self, *args, **kwargs):
        """Return traces error"""
        kwargs['yerr_only'] = True
        kwargs['y_only'] = False
        return self.trace(*args, **kwargs)

    def track(self, prop='basesubed', kwargs_prop=None, percent=False, aggr=np.nansum, **kwargs):
        """Time track.
        kwargs_prop: get passed to select as kwargs_prop,
        percecnt: if true, return is normalized to 100%
        additional kwargs are passed to select
        aggr: a function that takes(4d array, axis=pixel_index) as arguments.
          usually it is np.nansum, but it can also be np.median or np.nanmedian
          or anly of these aggregation functions from numpy.
          This function is used for aggregation over the pixel axis, so 1
          point per frame is generated.


        A time track is one number from each spectrum vs its recorded time point.
        """
        y = self.select(prop=prop, kwargs_prop=kwargs_prop, **kwargs)
        #y = np.nanmedian(y, axis=X_PIXEL_INDEX)
        y = aggr(y, axis=X_PIXEL_INDEX)
        if percent:
            y = y / y.max() * 100
        return y

    def time_domain(self, **kwargs):
        """
        Use Inverse-FFT to transform into time domain.

        prop default to basesubed

        **kwargs** get passed to *SfgRecod.select*
        """

        kwargs.setdefault('prop', 'basesubed')
        logger.debug('time_domain kwargs: {}'.format( kwargs ))
        ret = self.select(**kwargs)
        ret = fft.ifft(ret)
        return ret

    def frequency_domain(self, start, stop, shift=None, **kwargs):
        """
        Transform data into time_domain, then select region between start
        and stop and transform back into frequencie space.

        **start**: int
        **stop**: int
          start and stop are the cutoffs for the time_domain signal.
        shift: Shift the signal in frequency domain by shift in rad.
          A pi shift might be needed depending on the alignment.
        """
        #frame_med = None
        if isinstance(shift, type(None)):
            logger.debug('Using default shift, {}'.format( self.het_shift ))
            shift = self.het_shift

        time_domain = self.time_domain(**kwargs)
        if isinstance(start, type(None)) or isinstance(stop, type(None)):
            raise TypeError(
                'Must define start {} and stop {}'.format(start, stop)
            )
        # Filter out undesired times.
        filter_mask = np.zeros_like(time_domain)
        x = np.arange(filter_mask.shape[-1])
        filter_mask[:, :, :] = 1/(1+(np.exp((start-x)/0.0025)))-1/(1+(np.exp((stop-x)/0.0025)))
        time_domain *= filter_mask
        ret = fft.fft(time_domain)
        # Shifts the signal by value of shift in radiant.
        # `SfgRecord.shift` is default
        if not isinstance(shift, type(None)):
            if hasattr(shift, '__iter__'):
                frames = kwargs.get('roi_frames', slice(None))
                shift = shift[frames]
                if len(shift) < ret.shape[FRAME_AXIS_INDEX]:
                    msg = 'Not enough shift values given\nData {} \nShift: {}'
                    raise IndexError(msg.format(
                        ret.shape[FRAME_AXIS_INDEX], np.shape(shift))
                    )
                for j in range(ret.shape[FRAME_AXIS_INDEX]):
                    ret[:, j] = ret[:, j] * np.exp(1j * np.pi * shift[j])
                    logger.debug('Shifting with: {}'.format( np.exp(1j * np.pi * shift[j]) ))
            else:
                ret = ret * np.exp(1j * np.pi * shift)
                logger.debug('Shifting with: {}'.format( np.exp(1j * np.pi * shift) ))

        return ret

    def norm_het(self, **kwargs):
        """Get heterodyne signal of norm.

        **kwargs**
          start: start time during time selection
            defaults to `SfgRecords.norm_het_start`
          stop: stop time during time  selection
            defaults to `SfgRecords.norm_het_stop`
          shift: phase schift in radiants
            defaults to `SfgRecords.norm_het_shift`
        """
        kwargs['prop'] = 'norm'
        # Issue of self.norm_het_shift is none, then self.het_shift gets used
        kwargs.setdefault('shift', self.norm_het_shift)
        kwargs.setdefault('start', self.norm_het_start)
        kwargs.setdefault('stop', self.norm_het_stop)
        #kwargs.setdefault('frame_med', True)
        norm = self.frequency_domain(**kwargs)
        return norm

    def signal_het(self, shift=None, **kwargs):
        """Default heterodyne unnormalized signal."""
        kwargs.setdefault('start', self.het_start)
        kwargs.setdefault('stop', self.het_stop)
        if not isinstance(shift, type(None)):
            logger.debug('setting shift: {}'.format( shift ))
            kwargs['shift'] = shift
        return self.frequency_domain(**kwargs)

    def normalize_het(self, kwargs_frequency_domain=None, kwargs_norm_het=None,
                      shift=None, **kwargs):
        """Normalize heterodyne SFG measurment.

        kwargs_frequency_domain:
          dict with at least {'start': int, 'stop': int}. This dict is used to
          construct the real and imag part of the chi2 signal.
        shift: Shift the signal in units of radiant after normalization
        frame_med: Calculate frame mean after normalization
        **kwargs**: Get passes to both, the norm and the signal.
        """
        if not kwargs_frequency_domain:
            kwargs_frequency_domain = {}
        if not kwargs_norm_het:
            kwargs_norm_het = {}
        signal = self.signal_het(**kwargs_frequency_domain, **kwargs)
        norm = self.norm_het(**kwargs_norm_het, **kwargs)
        logger.debug('signal shape: {}'.format( signal.shape ))
        logger.debug('Norm Shape: {}'.format( norm.shape ))
        chi2 = signal/norm

        self.chi2 = chi2
        return chi2

    def trace_multiple(
            self,
            prop='bleach',
            kwargs_prop={'opt': 'rel', 'prop': 'basesubed'},
            roi_wavenumbers=[None],
            roi_delays=[None],
            shift_neg_time=False,
            **kwargs
    ):
        """Return multiple traces at the same time."""

        return [self.trace(prop, kwargs_prop, roi_wavenumber, roi_delay, shift_neg_time, **kwargs) for roi_wavenumber, roi_delay in zip(roi_wavenumbers, roi_delays)]

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

    def _readData(self, fname):
        """Read Data of Record from fname.

        **Arguments:**
          fname: path to data file or files.
        """

        imported = import_data(fname)
        self.type = imported['type']
        self.metadata = imported['metadata']
        if self.type == "spe":
            sps = imported['data']
            rawData = np.concatenate([sp.data for sp in sps], 0)
            self.rawData = rawData.reshape(1, *rawData.shape)
            if not self._setted_wavelength:
                self.wavelength = sps[0].wavelength
            try:
                self.calib_poly = sps[0].calib_poly
            except AttributeError:
                pass
            # TODO Update internal properties

        elif self.type == "npz":
            imp = imported['data']
            # This should be a bad idear, because it means
            # load and save function must be the same at all time
            for key, value in self.saveable.items():
                if key in imp.keys():
                    # There are some male formatted arrays
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

        elif self.type == "veronica":
            self.rawData, self.pp_delays = imported['data']

        elif self.type == "victor":
            self.rawData, self.pp_delays = imported['data']

        # Update datadependent rois
        if isinstance(self.roi_spectra, type(slice(None))):
            if not self.roi_spectra.stop:
                self.roi_spectra = slice(
                    self.roi_spectra.start,
                    self.number_of_spectra
                )
        if not self.roi_frames.stop:
            self.roi_frames = slice(self.roi_frames.start,
                                    self.number_of_frames)
        if not self.roi_delay.stop:
            self.roi_delay = slice(self.roi_delay.start,
                                   self.number_of_pp_delays)

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
        ret.rawData = self.rawData.copy()
        ret.base = self.base.copy()
        ret.norm = self.norm.copy()
        ret.pp_delays = self.pp_delays.copy()
        ret.metadata = self.metadata.copy()
        ret._fname = self._fname
        # For spe this leads to issues, because spe has the _spe attribute
        # This doesnt gets copied, so we have to introduce a new type
        # for copied spe files, that gets treated slightly different then
        # the original spe type. Otherwise we would need to copy _spe attribute
        # witch is even worse I think.
        ret.type = self.type
        if self.type == 'spe':
            ret.type = 'spe_c' # copied spe type
        ret._unpumped_index = self._unpumped_index
        ret._pumped_index = self._pumped_index
        ret.baseline_offset = self.baseline_offset
        ret.zero_time_select = self.zero_time_select
        ret.zero_time_subtraction = self.zero_time_subtraction
        ret.baseline_offset = self.baseline_offset
        ret.roi_x_pixel_spec = self.roi_x_pixel_spec
        ret.rois_x_pixel_trace = self.rois_x_pixel_trace
        ret.roi_spectra = self.roi_spectra
        ret.roi_frames = self.roi_frames
        ret.roi_delay = self.roi_delay
        ret.rois_delays_pump_probe = self.rois_delays_pump_probe
        ret.het_shift = self.het_shift
        ret.norm_het_shift = self.norm_het_shift
        if self._setted_wavenumber:
            ret.wavenumber = self.wavenumber
        if self._setted_wavelength:
            ret.wavelength = self.wavelength
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
        kwargs = {key: getattr(self, value) for key, value in self.saveable.items()}
        np.savez_compressed(
            file,
            **kwargs
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


def get_Record2d_from_sfgRecords(records):
    """Function to create a Record2d object from a list of SfgRecords.

    Each record must be recorded at a different pump frequencie.
    """
    number_of_records = len(records)
    pump_freqs = np.zeros(number_of_records)
    pumped = np.zeros((
        records[0].number_of_pp_delays,
        len(pump_freqs),
        records[0].number_of_x_pixel
    ))
    unpumped = np.zeros_like(pumped)
    pumpedE = np.zeros_like(pumped)
    unpumpedE = np.zeros_like(pumped)
    wavenumber = records[0].wavenumber
    pp_delays = records[0].pp_delays
    for i in range(number_of_records):
        record = records[i]
        pump_freqs[i] = record.pump_freq
        pumped[:, i] = record.pumped(
            frame_med=True
        )[:, 0 , 0]
        unpumped[:, i] = record.unpumped(
            frame_med=True
        )[:, 0 , 0]
        pumpedE[:, i] = record.sem(
            'pumped',
            roi_pixel=slice(None)
        )[:, 0]
        unpumpedE[:, i] = record.sem(
            'unpumped',
            roi_pixel=slice(None)
        )[:, 0]
        if not np.all(wavenumber == record.wavenumber):
            msg = 'Wavenumbers in 2d Spectra must all be the same\nIssue in {}'.format(i)
            raise IOError()
        if not np.all(pp_delays == record.pp_delays):
            msg = 'Delays in 2d Spectra must all be the same.\nIssue in {}'.format(i)
            raise IOError()

    return Record2d(
        pump_freqs,
        pumped,
        unpumped,
        pp_delays,
        wavenumber,
        pumpedE,
        unpumpedE,
       )


def load_Record2d(fname):
    """Load Record2d from a .npz file"""

    data = np.load(fname)
    pump_freqs = data['pump_freqs']
    pumped = data['pumped']
    unpumped = data['unpumped']
    ret = Record2d(pump_freqs, pumped, unpumped)
    for key, value in data.items():
        setattr(ret, key, value)
    return ret


class Record2d():
    def __init__(
            self,
            pump_freqs,
            pumped,
            unpumped,
            pp_delays=None,
            wavenumbers=None,
            pumpedE=None,
            unpumpedE=None,
            gold_bleach=None,
            static_chi=None,
            zero_time_offset=None,
    ):
        """Record of 2D data.

        Indexing is of the form of:
        [pp_delays, pump_freqs, x-pixels]

        **Arguments:**
          - **pump_freqs**: Array of pumped frequencies
          - **pumped**: 3d Array of pumped data
          - **unpumped**: 3d Array of unpumped data

        **Keywords:**
          - **pp_delays**: Array of pump_probe delays
          - **wavenumbers**: Array of wavenumbers
          - **pumpedE**: 3d array of pumped data uncertaincy
          - **unpumpedE**: 3d array of unpumped data uncertaincy
          - **gold_bleach**: array of minimal bleach values on gold
          - **static_chi**: array of correction factors for static Ampitude
          - **zero_time_offset**: array of offset values for the relative zero
              time correction. This correction applyes, because the 0 time is
              necessary the same throughout multiple measurments.
        """


        self.pumped = pumped
        self.unpumped = unpumped
        self.pump_freqs = pump_freqs

        #self.pp_delays = np.ones_like(self.pumped[0:1]) * np.arange(
        #    self.pumped.shape[0])
        if not isinstance(pp_delays, type(None)):
            self.pp_delays = pp_delays
        self.wavenumbers = np.ones_like(self.pumped) * np.arange(
            self.pumped.shape[-1], 0, -1
        )
        if not isinstance(wavenumbers, type(None)):
            self.wavenumbers = wavenumbers

        self.pumpedE = np.zeros_like(self.pumped)
        if not isinstance(pumpedE, type(None)):
            self.pumpedE = pumpedE

        self.unpumpedE = np.zeros_like(self.unpumped)
        if not isinstance(unpumpedE, type(None)):
            self.unpumpedE = unpumpedE

        self.zero_time_subtraction = True
        self.zero_time_select = slice(0, 2)

        self.gold_bleach = gold_bleach
        self.static_chi = static_chi

    @property
    def number_of_delays(self):
        return self.pp_delays.shape[0]

    @property
    def number_of_pump_freqs(self):
        return self.pump_freqs.shape[0]

    @property
    def pp_delays_corrected(self):
        """Corrected version of pp_delays.

        Uses `Record2d.zero_time_offset` to calculate an effective delay times matrix.
        This  matrix has shape of (number_of_pump_freqs, number_of_pp_delays)
        """
        times = np.ones((self.number_of_pump_freqs, self.number_of_delays)) * self.pp_delays
        corrections = np.transpose(np.ones_like(times).T*self.zero_time_offset)
        ctimes = times+corrections
        return ctimes.astype(int)

    def find_delay_index(self, value):
        """Find closest pp_delay index for value

        **Arguments:**
          - **value**: number of delay time to search for in `Record2d.pp_delays_corrected`

        **Returns:**
          Array of index positions, that effectively match the value."""
        return abs(self.pp_delays_corrected-value).argmin(1)

    def bleach(self, opt='rel', **kwargs):
        pumped = self.select('pumped', **kwargs)
        unpumped = self.select('unpumped'**kwargs)

        if opt is 'rel':
            bleach = pumped / unpumped
        elif opt is 'abs':
            bleach = pumped - unpumped
        else:
            raise ValueError(
                "Must enter valid opt {} is invalid".format(opt)
            )
        if self.zero_time_subtraction:
            zero_time = bleach[self.zero_time_select].mean(0)
            bleach -= zero_time
            # Recorretion for zero_time offset needed because
            # data is expected to be at 1 for negative times.
            if opt is 'rel':
                bleach += 1
                self.zero_time_rel = zero_time
            else:
                # TODO add the mean of zero_time
                self.zero_time_abs = zero_time

        # Correct infs and nans.
        np.nan_to_num(bleach, copy=False)
        return bleach

    def static(
            self,
            delay,
            prop='unpumped',
            roi_pixel=slice(None),
            scale=1,
            medfilt_kernel=None,
            resample_freqs=0,
    ):
        """Data for a static spectrum."""
        y = getattr(self, prop)[delay, :, roi_pixel].T
        if medfilt_kernel:
            y = medfilt(y, medfilt_kernel)
        if resample_freqs:
            y = double_resample(y, resample_freqs)

        return scale*y

    def pump_vs_probe(
            self,
            delay,
            prop='bleach',
            kwargs_prop=None,
            roi_pixel=slice(None),
            medfilt_kernel=None,
            resample_freqs=0,
            norm_by_gold_bleach=False,
            norm_by_static_spectra=False,
            shift_zero_time_offset=False,
            scale=1,
    ):
        """Data for a contour plot with pump vs probe.

        **Arguments:**
          - **delay**: Index of pump_probe delay to get data from.

        **Keywords**:
          - **prop**: Default 'bleach'. Property of data.
          - **kwargs_prop**: Keywords to select property with.
          - **roi_pixel**: Pixel Region of interest of data
          - **medfilt_kernel**: Median filter kernel. Two element array with:
              (probe, pump)
          - **resample_freqs**: Frequencies for the resample filter
          - **norm_by_gold_bleach**: Normalize with Record2d.gold_bleach
          - **norm_by_static_spectra**: Normalizes with static amplitudes
          - **shift_zero_time_offset**: Shift zero time offset. This matches
              bleach by time.
          - **scale**: Scale result z axis

        **Returns:**
        2d Numpy array with pump vs probe orientation.

        """
        if not kwargs_prop:
          kwargs_prop = {}
        z_raw = getattr(self, prop)(**kwargs_prop)
        if shift_zero_time_offset:
            time = self.pp_delays[delay]
            best_delay_indeces = self.find_delay_index(time)
            logger.debug('Combining {} as {}'.format(time, self.pp_delays[self.find_delay_index(time)]))
            z = z_raw[best_delay_indeces, range(self.number_of_pump_freqs), roi_pixel].T
        else:
            z = z_raw[delay, :, roi_pixel].T
        opt = kwargs_prop.get('opt', 'rel')
        if medfilt_kernel:
            z = medfilt(z, medfilt_kernel)
        if resample_freqs:
            z = double_resample(z, resample_freqs)
        if norm_by_gold_bleach:
            if opt == 'rel':
                z = (z-1) * (1+self.gold_bleach) + 1
            elif opt == 'abs':
                z = z * (1 + self.gold_bleach)
            else:
                raise NotImplementedError
        if norm_by_static_spectra:
            if opt == 'rel':
                z = (z - 1) * self.static_chi + 1
            elif opt == 'abs':
                z = z / self.static_chi
            else:
                raise NotImplementedError
        return scale*z


    def save(self, file):
        """Save Record2d into a numpy array obj."""
        kwargs = dict(
            pump_freqs=self.pump_freqs,
            pumped=self.pumped,
            unpumped=self.unpumped,
            pp_delays=self.pp_delays,
            wavenumbers=self.wavenumbers,
            pumpedE=self.pumpedE,
            unpumpedE=self.unpumpedE,
            gold_bleach=self.gold_bleach,
            static_chi=self.static_chi,
            zero_time_offset=self.zero_time_offset
        )
        logger.debug('Saving to {}'.format(path.abspath(file)))
        np.savez_compressed(
            file,
            **kwargs
        )


def concatenate_list_of_SfgRecords(list_of_records):
    """Concatenate SfgRecords into one big SfgRecord.

    Uses records.concatenate_list_of_SfgRecord. Is kept here for backwards
    compatibility.
    """
    logger.warn('Method deprecated. Use from records module instead.')
    from records import concatenate_list_of_SfgRecords
    return concatenate_list_of_SfgRecords(list_of_records)


def SfgRecords_from_file_list(list, **kwargs):
    """Import a list of files as a single SfgRecord.

    list: list of filepaths to import SfgRecords from.
    """
    return concatenate_list_of_SfgRecords([SfgRecord(elm, **kwargs) for elm in list])

def get_fit_results(record):
    """Extract fit results from a record with a minuit based model fit.

    **Arguments:**
      - **record**: The Record to extract the results from

    **Returns:**
    List of fit results. Each fit restuls is a dictionary with information
    about the fit.
    """
    ret = []
    for binning, model in record.models.items():
        ret.append(
            {
                'fitarg': model.minuit.fitarg,
                'name': record.name,
                'pump_freq': record.pump_freq,
                'binning': binning,
            }
        )
    return ret
