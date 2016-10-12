from . import veronica #import read_save, read_scan_stack
from . import spe

from os import path
from numpy import genfromtxt, array, delete, zeros


class AllYouCanEat():
    def __init__(self, fname):
        if not path.isfile(fname) and \
           not path.islink(fname):
            raise IOError('%s does not exist' % fname)

        self._fname = path.abspath(fname)
        
        self.readData()

    def readData(self):
        self._get_type()
        self._arange_data()
        #self._read_metadata()

    def _get_type(self):
        self._get_type_by_path()
        self._check_type()

    def _arange_data(self):
        if self._type == "spe":
            self._arange_spe()
            return

        if self._type == "sp":
            self._arange_sp()
            return

        if self._type == "ts" or self._type == 'sc':
            self._arange_ts()
            return

        raise NotImplementedError("Uuuups this should never be reached."
                                  "Bug with %s" % self._fname)

    def _read_metadata(self):
        pass

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

        print("type by name and type by shape don't match for %s" %self._fname)
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

        raise NotImplementedError("Uuups this should never be reached."
                                  "Shape of %s is %s" % (self._fname, self._data.shape))

    def _arange_spe(self):
        sp = spe.PrincetonSPEFile3(self._fname)
        self._data = sp.data.reshape(1, sp.NumFrames, sp.ydim, sp.xdim)
    
    def _arange_sp(self):
        """Makes a scan having the same data structure as spe """
        # 3 because victor saves 3 spectra at a time
        # 1 because its only 1 spectrum no repetition
        # x axis given by the pixels
        self._data = self._data[:,1:4].reshape(1, 1, 3, veronica.PIXEL)

    def _arange_ts(self):
        self._data
        def _iterator(elm):
            return array(
                # ditch first iteration because it the AVG
                # and correct NumReps for the missing AVG entry
                [elm(i) for i in range(1, self.NumReps + 1)]
            ).flatten().tolist()

        # Remove unneeded columns
        colum_inds = [0] + _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
        self._data = self._data[:,colum_inds]
        
        # pop pp_delays from data array and remove rows
        pp_delays = self._data[0::1602, 0]
        self._data = delete(self._data, slice(0, None, 1602), 0)

        # Now We know the rest is uint32
        self._data = self._data.astype('uint32')

        # pop empty rows
        empty_rows = [i*1600 + i-1 for i in range(self._data.shape[0]//1600)]
        self._data = delete(self._data, empty_rows, 0)

        # Currently numpy ignores the [-1, ...] in the empty_rows, but we want
        # the last line to be deleted as well, because it is empty
        if self._data.shape[0]%1600 == 1:
            self._data = delete(self._data, -1, 0)

        # Remove the 0 (pixel) column it is redundant
        self._data = self._data[:, 1:]

        # We want data to fit in the same structure as all the other scans
        ret = zeros((self.NumPp_delays, self.NumReps, veronica.SPECS, veronica.PIXEL))
        for i_row in range(len(self._data)):
            row = self._data[i_row]
            # The index of the current delay
            i_delay = i_row // veronica.PIXEL
            i_pixel = i_row % veronica.PIXEL
            for i_col in range(len(row)):
                # The index of the current repetition
                i_rep = i_col // self.NumReps
                i_spec = i_col % veronica.SPECS # 3 is the number of specs
                ret[i_delay, i_rep, i_spec, i_pixel] = row[i_col]

        self._data = ret
        
        
