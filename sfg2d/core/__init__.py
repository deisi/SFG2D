from os import path
import warnings

import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from . import scan
from ..io.veronica import SPECS, pixel_to_nm, get_from_veronika
from ..io.victor_controller import get_from_victor_controller
from ..utils import nm_to_ir_wavenumbers, X_PIXEL_INDEX, Y_PIXEL_INDEX, \
    SPEC_INDEX, FRAME_AXIS_INDEX, PIXEL, PP_INDEX, X_PIXEL_INDEX

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
    def __init__(self, fname=None, base=None, norm=None):
        self.pp_delays = np.array([0])
        self.metadata = {}
        if not fname:
            self._fname = ""
        else:
            self._fname  = fname
        self._rawData = np.zeros((1, 1, 1, PIXEL))
        self._data = self._rawData
        self._base = np.zeros_like(self.rawData)
        self._norm = np.ones_like(self.rawData)
        self.isBaselineSubed = False
        self._basesubed = None
        self.isNormalized = False
        self._normalized = None
        self._type = 'unknown'
        self._wavelength = None
        self._wavenumber = None
        self._bleach = None
        self._bleach_rel = None
        self._pumped = None
        self._unpumped = None
        self._pumped_index = 0
        self._unpumped_index =  1
        self._dates = None
        self.zero_time = None
        self.zero_time_rel = None

        if isinstance(fname, type(None)):
            return

        if '~' in fname:
            fname = path.expanduser(fname)
        self._fname = path.abspath(fname)
        self._dates = None

        if not isinstance(base, type(None)):
            if isinstance(base, str):
                base = SfgRecord(base).data
                self.base = base
            else:
                self.base = base

        if not isinstance(norm, type(None)):
            if isinstance(norm, str):
                norm = SfgRecord(norm).data
                self.norm = norm
            else:
                self.norm = norm

        self._readData()

    @property
    def rawData(self):
        return self._rawData

    @rawData.setter
    def rawData(self, value):
        if not isinstance(value, np.ndarray):
            raise IOError("Can't use type %s for data" % type(value))
        if len(value.shape) != 4:
            raise IOError("Can't set shape %s to data" % value.shape)
        self._rawData = value
        self._data = self._rawData
        self.isBaselineSubed = False

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        if self.isBaselineSubed:
            self.add_base()
            self.isBaselineSubed = False
        self._base = value * np.ones_like(self.rawData)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        if self.isNormalized:
            self.un_normalize()
            self.isNormalized = False
        self._norm = value * np.ones_like(self.rawData)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
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
        time_of_a_scan = self.metadata['exposure_time']
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[0]):
                ret.append(
                    (i * self.number_of_pp_delays + j) * time_of_a_scan
                )
        return ret

    @property
    def frames(self):
        """Number of frames."""
        warnings.warn(
            "SfgRecord.frames is deprecated."\
            " Use SfgRecord.numer_of_frames instead"
        )
        return self.data.shape[FRAME_AXIS_INDEX]

    @property
    def number_of_frames(self):
        """Number of frames"""
        return self.data.shape[FRAME_AXIS_INDEX]

    @property
    def pixel(self):
        return np.arange(self.data.shape[X_PIXEL_INDEX])

    @property
    def number_of_y_pixel(self):
        return self.data.shape[Y_PIXEL_INDEX]

    @property
    def number_of_spectra(self):
        return self.number_of_y_pixel

    @property
    def number_of_x_pixel(self):
        return self.data.shape[X_PIXEL_INDEX]

    @property
    def number_of_pp_delays(self):
        return self.data.shape[PP_INDEX]

    @property
    def wavelength(self):
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
    def wavenumber(self):
        """wavenumber propertie."""

        if isinstance(self._wavenumber, type(None)):
            vis_wl = self.metadata.get("vis_wl")
            self._wavenumber = self.get_wavenumber(vis_wl)
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, arg):
        """Setter for wavenumber propertie."""

        self._wavenumber = arg

    def get_wavenumber(self, vis_wl):
        """return calculated wavenumbers."""

        ret = self.pixel[::-1]
        if isinstance(vis_wl, type(None)) or vis_wl < 1:
            return ret
        ret = nm_to_ir_wavenumbers(self.wavelength, vis_wl)
        return ret

    def reset_data(self):
        """reset `SfgRecord.data` to `SfgRecord.rawData`"""

        self.data = self.rawData
        self.isBaselineSubed = False
        self.isNormalized = False

    @property
    def basesubed(self):
        """Baselinesubstracted data set."""

        if isinstance(self._basesubed, type(None)):
            self._basesubed = self.get_baselinesubed()
        return self._basesubed

    def get_baselinesubed(self, use_rawData=True):
        """Get baselinesubstracted data"""

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
            Subtract baseline from `SfgRecord.rawData` and not `SfgRecord.data`.
            If data is already normalized (`SfgRecord.isNormalized` is True) then,
            this is always calculated from `SfgRecord.rawData`. So be carefull about
            this when using filters, because they usually only work on `SfgRecord.data`
            and not `SfgRecord.rawData`.

        Returns
        -------
        Numpy array with SfgRecod.data subsituted by SfgRecord.base"""
        ret = self.get_baselinesubed(use_rawData)
        if inplace and not self.isBaselineSubed:
            self.data = ret
            # Toggle to prevent multiple baseline substitutions
            self.isBaselineSubed = True
        return ret

    def add_base(self, inplace=False):
        """Add baseline to data"""
        ret = self.data + self.base
        if inplace and not self.isBaselineSubed:
            self.data = ret
            self.isBaselineSubed = False
        return ret

    @property
    def normalized(self):
        """Normalized data set"""
        if isinstance(self._normalized, type(None)):
            self._normalized = self.get_normalized()
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

    def normalize(self, inplace=False, use_rawData=False):
        if not self.isBaselineSubed:
            warnings.warn(
                "Normalizing data although baseline is not substracted."\
                "Consider subtracting baseline with `SfgRecod.sub_base(inplace=True)`"\
                "first."
            )
        ret = self.get_normalized(use_rawData)
        if inplace and not self.isNormalized:
            self.data = ret
            self.isNormalized = True
        return ret

    def un_normalize(self, inplace=False):
        if not self.isBaselineSubed:
            warnings.warn(
                "Normalizing data although baseline is not substracted."\
                "Consider subtracting baseline with `SfgRecod.sub_base(inplace=True)`"\
                "first."
            )
        ret = self.data * self.norm
        if inplace and self.isNormalized:
            self.data = ret
            self.isNormalized = False
        return ret

    @property
    def sum_argmax(self, frame_median=True, pixel_slice=slice(None,None)):
        """argmax of pp_delay with biggest sum.

        frame_median : bool
            default true, Take median of frames.
            if false give a frimewise argmax."""
        if frame_median:
            ret = np.median(self.data[:,:,:,pixel_slice], 1).sum(-1).argmax(0)
        else:
            ret = self.data[:,:,:,pixel_slice].sum(-1).argmax(0)
        return ret

    @property
    def pumped(self):
        if isinstance(self._pumped, type(None)):
            self.pumped = self._pumped_index
        return self._pumped

    @pumped.setter
    def pumped(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self._pumped = self.data[:, :, value]
            self._pumped_index = value
        else:
            raise NotImplemented()

    @property
    def unpumped(self):
        if isinstance(self._unpumped, type(None)):
            self.unpumped = self._unpumped_index
        return self._unpumped

    @unpumped.setter
    def unpumped(self, value):
        if isinstance(value, int) or isinstance(value, np.integer):
            self._unpumped = self.data[:, :, value]
            self._unpumped_index = value
        else:
            raise NotImplemented()

    @property
    def bleach(self):
        if isinstance(self._bleach, type(None)):
            self._bleach = self.get_bleach()
        return self._bleach

    def get_bleach(self, pumped=None, unpumped=None, sub_first=True):
        """Calculate bleach.

        sub_first : boolean
            substitute the first spectrum from the data to account for
            the constant offset."""
        #TODO catch the case when there is only one spectrum

        # Init needed to prevent inplace overwriting of self.data
        bleach = np.zeros((
            self.number_of_pp_delays,
            self.number_of_spectra,
            self.number_of_x_pixel,
        ))

        if self.number_of_spectra < 2:
            return bleach

        if isinstance(pumped, type(None)):
            pumped = self.pumped
        else:
            # Uses the setter of SfgRecord.pumped to recast pumped to an array
            self.pumped = pumped
            pumped = self.pumped
        if isinstance(unpumped, type(None)):
            unpumped = self.unpumped
        else:
            self.unpumped = unpumped
            unpumped = self.unpumped

        bleach = pumped - unpumped
        if sub_first:
            # copy needed to prevent inplace overwriting of bleach
            self.zero_time = bleach[0].copy()
            bleach -= self.zero_time
        return bleach

    @property
    def bleach_rel(self):
        if isinstance(self._bleach_rel, type(None)):
            self._bleach_rel = self.get_bleach_rel()
        return self._bleach_rel

    def get_bleach_rel(self, pumped=None, unpumped=None, sub_first=True):
        """Calculate the relative bleach

        Parameters
        ----------
        pumped: index or None
            y_pixel/spectra index of the pumped spectrum. Default is None,
            and will make it use `SfgRecord._pumped_index`
        unpumped: index or None
            like pumped only for unpumped None defaults to `SfgRecord._unpumped_index`
        sub_first: bollean default true
            subtract the 0th pp_delay index. This corrects for constant
            offset between pumped and unpumped data.
        """
        bleach_rel = np.zeros((
            self.number_of_pp_delays,
            self.number_of_spectra,
            self.number_of_x_pixel,
        ))

        if self.number_of_spectra < 2:
            return bleach

        if isinstance(pumped, type(None)):
            pumped = self._pumped_index
        if isinstance(unpumped, type(None)):
            unpumped = self._unpumped_index

        bleach_rel = self.data[:, :, pumped] / self.data[:, :, unpumped]
        if sub_first:
            self.zero_time_rel = bleach_rel[0].copy()
            bleach_rel -= self.zero_time_rel
        return bleach_rel

    @property
    def trace_pp_delay(self):
        return self.get_trace_pp_delay()

    def get_trace_pp_delay(
            self,
            frame_slice=slice(None),
            y_pixel_slice=slice(None),
            x_pixel_slice=slice(None),
            frame_median=True,
            medfilt_kernel=None,
    ):
        """Calculate the pp_delay wise trace."""

        ret = self.data[:, frame_slice, y_pixel_slice, x_pixel_slice]
        if isinstance(medfilt_kernel, tuple):
            ret = medfilt(ret, medfilt_kernel)
        if frame_median:
            ret = np.median(ret, FRAME_AXIS_INDEX)
        return ret.sum(X_PIXEL_INDEX)

    @property
    def trace_frame(self):
        return self.get_trace_frame()

    def get_trace_frame(
            self,
            pp_delay_slice=slice(None),
            y_pixel_slice=slice(None),
            x_pixel_slice=slice(None),
            pp_delay_median=True,
            medfilt_kernel=None
    ):
        ret = self.data[pp_delay_slice, :, y_pixel_slice, x_pixel_slice]
        if isinstance(medfilt_kernel, tuple):
            ret = medfilt(ret, medfilt_kernel)
        if pp_delay_median:
            ret = np.median(ret, PP_INDEX)
        return ret.sum(X_PIXEL_INDEX)

    def median(self, ax=None):
        """Calculate the median for the given axis.

        np.median(self.data, ax) is calculated."""
        return np.median(self.data, ax)

    def get_linear_baseline(self, start_slice=None, stop_slice=None, data_attr="rawData"):
        """Calculate a linear baseline from data"""
        data  = getattr(self, data_attr)

        if isinstance(start_slice, type(None)):
            start_slice = slice(0, 0.1*self.pixel)

        if isinstance(stop_slice, type(None)):
            stop_slice = slice(-0.1*self.pixel, None)

        yp = np.array([np.median(data[:,:,:,start_slice], -1), np.median(data[:,:,:,stop_slice], -1)])
        xp = [0, self.pixel]
        x = np.arange(self.pixel)
        #TODO Make use of gridspec or any other multitimensional linear interpolation method.
        raise NotImplementedError

    def get_trace_bleach(
            self, attr="bleach", pp_delay_slice=slice(None), frame_slice=slice(None),
            x_pixel_slice=slice(None), medfilt_kernel=None,
    ):
        ret = getattr(self, attr)
        if not isinstance(medfilt_kernel, type(None)):
            ret = medfilt(ret, medfilt_kernel)

        ret = ret[pp_delay_slice, frame_slice, x_pixel_slice].sum(-1)

        return ret

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
        if path.splitext(ftail)[1] ==  '.npz':
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
                    if  (start_of_data.shape[1] - 1) % 3 == 0:
                        self._type = 'victor'
                        return True
                else:
                    if start_of_data.shape[1]%6 == 0:
                        self._type = 'veronica'
                        return True
        raise IOError("Cant understand data in %f" % self._fname)

    def _import_data(self):
        """Import the data."""

        # Pick import function according to data type automatically.
        if self._type == "spe":
            from ..io.spe import PrincetonSPEFile3
            self._sp = PrincetonSPEFile3(self._fname)
            self.rawData = self._sp.data.reshape(1, self._sp.NumFrames, self._sp.ydim, self._sp.xdim)
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
            return

        if self._type == "veronica":
            self.rawData, self.pp_delays = get_from_veronika(self._fname)
            return

        if self._type == "victor":
            self.rawData, self.pp_delays = get_from_victor_controller(self._fname)
            return

        raise NotImplementedError("Uuuups this should never be reached."
                                  "Bug with %s. I cannot understand the datatype" % self._fname
                                  )

    def _read_metadata(self):
        """Read metadata of the file"""

        from ..utils.metadata import get_metadata_from_filename

        if self._type == "npz":
            return

        try:
            metadata = get_metadata_from_filename(self._fname)
        #TODO Refactor this, if I would program better this
        # would not happen
        except ValueError:
            warnings.warn('ValueError while trying to extract metadata from filepath./nSkipping')

        if self._type == "victor":
            # Read metadata from file header.
            from ..io.victor_controller import read_header, translate_header_to_metadata
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
        return ret

    def save(self, file, *args, **kwargs):
        """Save the SfgRecord into a compressed numpy array."""
        np.savez_compressed(
            file,
            rawdata = self.rawData,
            norm = self.norm,
            base = self.base,
            pp_delays = self.pp_delays,
            wavelength = self.wavelength,
            wavenumber = self.wavenumber,
            metadata = self.metadata,
            dates = self.dates,
            _pumped_index = self._pumped_index,
            _unpumped_index = self._unpumped_index
        )

    def make_avg(self):
        """Returns an frame wise averaged SfgRecord."""
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
        return ret

    def plot(self,
            pp_delays=slice(None,None), frames=slice(None, None),
            y_pixel=slice(None, None), x_pixel=slice(None, None),
             attribute="data", frame_med_slice=slice(None, None),
            fig=None, ax=None, x_axis="pixel", filter_kernel=(1,1,1,11), **kwargs):
        """Plot the SfgRecord.

        Parameters:
        -----------
        pp_delays: slice or array
            index of interesting pp_delays
        frames: slice or array
            index of relevant frames
        y_pixel: slice or array
            index of spectra of y_pixel
        x_pixel: slice or array
            index of x_pixel
        attribute: string
            attribute of SfgRecord to plot. Default is 'data'
        frame_med_slice: slice or array
            not impltemented yet
        fig: matplotlib figure obj,
        x_axis: string
            name of the x_axis parameter to use. Default is "pixel"
        filter_kernel: 4d tuple or list
            kernel of the med_filter function. default is (1,1,1,11)
            meaning filter 11 pixel and leve the rest untouched.
        """


        if not fig:
            fig = plt.gcf()

        if not ax:
            ax = plt.gca()

        if isinstance(x_axis, str):
            x_axis = getattr(self, x_axis)

        data = getattr(self, attribute)

        plot_data = medfilt(
            data, filter_kernel
        )[pp_delays, frames, y_pixel, x_pixel]

        lines = []
        for per_delay in plot_data:
            frame_lines = []
            for per_frame in per_delay:
                frame_lines.append(ax.plot(x_axis, per_frame.T, **kwargs))
            lines.append(frame_lines)

        return lines

    def plot_bleach(self, attribute="bleach", filter_kernel=(1,1,11),
                    pp_delays=slice(None,None), frames=slice(None,None),
                    x_pixel=slice(None, None), fig=None, ax=None,
                    x_axis="pixel",
                    **kwargs):
        """Called when attribute is 'bleach' in plot."""
        if not fig:
            fig = plt.gcf()

        if not ax:
            ax = plt.gca()

        if isinstance(x_axis, str):
            x_axis = getattr(self, x_axis)

        data = getattr(self, attribute)

        plot_data = medfilt(
            data, filter_kernel
        )[pp_delays, frames, x_pixel]

        lines = []
        for per_delay in plot_data:
            frame_lines = []
            for per_frame in per_delay:
                frame_lines.append(ax.plot(x_axis, per_frame.T, **kwargs))
            lines.append(frame_lines)

        return lines

    def plot_trace(
            self,
            x_axis="pp_delays",
            fig = None, ax = None,
            **kwargs
    ):
        """

        kwargs are passes to SfgRecord.get_trace_pp_delays

        """
        if not fig:
            fig = plt.gcf()

        if not ax:
            ax = plt.gca()

        x_axis = getattr(self, x_axis)

        plot_data = self.get_trace_pp_delay(**kwargs)

        lines = ax.plot(x_axis, plot_data, "-o")

        return lines


def concatenate_SfgRecords(list_of_records):
    """Concatenate SfgRecords into one big SfgRecord."""
    ret = SfgRecord()
    ret.metadata["central_wl"] = None
    ret.wavelength = list_of_records[0].wavelength
    ret.rawData = list_of_records[0].rawData.copy()
    ret.rawData =  np.concatenate([elm.rawData for elm in list_of_records], 1)
    ret.dates = np.concatenate([elm.dates for elm in list_of_records]).tolist()
    ret.pp_delays = list_of_records[0].pp_delays
    ret.metadata["central_wl"] = list_of_records[0].metadata.get("central_wl")
    ret.metadata["vis_wl"] = list_of_records[0].metadata.get("vis_wl")
    return ret
