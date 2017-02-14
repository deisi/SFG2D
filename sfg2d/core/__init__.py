from os import path
import warnings

import numpy as np

from . import scan
from ..io.veronica import SPECS, pixel_to_nm, get_from_veronika
from ..io.victor_controller import get_from_victor_controller
from ..utils.static import nm_to_ir_wavenumbers
from ..utils.consts import X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX, FRAME_AXIS_INDEX, PIXEL, PP_INDEX

class SfgRecord():
    """Class to load and manage SFG data in a 4D structure

    It is ment to encapsulate raw SFG data in a unifying manner.
    For dimensions are needed to cope pump-probe delay, frames(repetitions),
    y-pixel and x-pixel.

    Reads
    -----
      - veronica .dat files
      - .spe files from Andor version 2.5 and version 3
      - victor_controller .dat files

    Properties
    ----------
    data : 4 dim numpy array
        Each axis seperates the data by:
        - axis 0: pump-probe time delay
        - axis 1: frames
        - axis 2: y-pixels
        - axis 3: x-pixels

    dates: list of dates
        each element of the list is a datetime obj refering to the time
        the data was recorded.

    times: list of timedeltas
        each element of the list is a relative timedelta obj, that refers
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
    """
    def __init__(self, fname=None):
        self.pp_delays = np.array([0])
        self.metadata = {}
        if not fname:
            self._fname = ""
        else:
            self._fname  = fname
        self._data = np.zeros((1,1,1,PIXEL))
        self._type = 'unknown'
        self._wavelength = None
        self._wavenumber = None
        self._bleach = None
        self._bleach_rel = None
        self.zero_time = None
        self.zero_time_rel = None

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
        for i in range(self.data.shape[1]):
            ret.append(
                i * self.metadata['exposure_time']
            )
        return ret

    @property
    def frames(self):
        """Number of frames."""
        return self.data.shape[FRAME_AXIS_INDEX]

    @property
    def pixel(self):
        return np.arange(self.data.shape[X_PIXEL_INDEX])

    @property
    def wavelength(self):
        if isinstance(self._wavelength, type(None)):
            self._wavelength = self.pixel
            cw = self.metadata.get('central_wl')
            #TODO check the cw is None case better
            if cw and cw >= 1:
                self._wavelength = pixel_to_nm(
                    np.arange(PIXEL),
                    cw,
                )
        return self._wavelength

    @wavelength.setter
    def wavelength(self, arg):
        self._wavelength = arg

    @property
    def wavenumber(self):
        if isinstance(self._wavenumber, type(None)):
            vis_wl = self.metadata.get("vis_wl", 1)
            if vis_wl == 1:
                self.metadata["vis_wl"] = 1
            self._wavenumber = nm_to_ir_wavenumbers(
                self.wavelength, vis_wl
            )
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, arg):
        self._wavenumber = arg

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
    def bleach(self):
        if isinstance(self._bleach, type(None)):
            self.calc_bleach()
        return self._bleach

    @property
    def bleach_rel(self):
        if isinstance(self._bleach_rel, type(None)):
            self.calc_bleach_rel()
        return self._bleach_rel

    def calc_bleach(self, pumped=0, unpumped=1, sub_first=True):
        """Calculate bleach.

        sub_first : boolean
            substitute the first spectrum from the data to account for
            the constant offset."""
        #TODO catch the case when there is only one spectrum
        self._bleach = self.data[:,:,pumped] - self.data[:, :, unpumped]
        if sub_first:
            self.zero_time = self._bleach[0].copy()
            self._bleach -= self.zero_time
        return self._bleach

    def calc_bleach_rel(self, pumped=0, unpumped=1, sub_first=True):
        """Calculate the relative bleach"""
        self._bleach_rel = self.data[:, :, pumped] / self.data[:, :, unpumped]
        if sub_first:
            self.zero_time_rel = self._bleach_rel[0].copy()
            self._bleach_rel -= self.zero_time_rel
        return self._bleach_rel


    def median(self, ax=None):
        """Calculate the median for the given axis.

        np.median(self.data, ax) is calculated."""
        return np.median(self.data, ax)

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

        # One needs to use the type, because we must pick the right import
        # function. Otherwise errors occur.
        if self._type == "spe":
            from ..io.spe import PrincetonSPEFile3
            self._sp = PrincetonSPEFile3(self._fname)
            self._data = self._sp.data.reshape(1, self._sp.NumFrames, self._sp.ydim, self._sp.xdim)
            return

        if self._type == "veronica":
            self._data, self.pp_delays = get_from_veronika(self._fname)
            return

        if self._type == "victor":
            self._data, self.pp_delays = get_from_victor_controller(self._fname)
            return

        raise NotImplementedError("Uuuups this should never be reached."
                                  "Bug with %s. I cannot understand the datatype" % self._fname
                                  )

    def _read_metadata(self):
        """Read metadata of the file"""

        from ..utils.metadata import get_metadata_from_filename

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

        self.metadata = metadata

    def copy(self):
        """Make a copy of the SfgRecord obj.

        This ensures, that we can have multiple disjoint objects, that
        come from the same file. By copying the data from the RAM, we can
        save some IO and speed up the process."""
        ret = SfgRecord()
        ret.pp_delays = self.pp_delays.copy()
        ret.metadata = self.metadata.copy()
        ret._fname = self._fname
        ret._data = self._data
        ret._type = self._type
        ret._dates = self._dates
        ret.wavelength = self._wavelength.copy()
        return ret
