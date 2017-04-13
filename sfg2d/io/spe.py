import struct
import numpy as np
import os
from datetime import datetime, timedelta

debug = 0

class PrincetonSPEFile3():
    """Class to import and read spe files.

    Capabilities 
    -----------
    Use this class to import spe files from Andor or Lightfield.
    It can handle spe v2 and spe v3 data.

    **Note**
    It only supports rectangular data, and is not feature complete.
    I only implemented what I actually need.

    Attributes
    ----------
    data : array
        (Number of frames, ydim, ydim) shaped array of the raw data

    wavelength : array
        Wavelength as it was given by the camera and the spectrometer

    central_wl : float
        central wavelength of the spectrometer

    headerVersion : float
        Version number of the header.

    xdim : int
        Number of pixels in x direction

    ydim : int
        Number of pixels in y direction

    NumFrames : int
        Number of Frames. I.e. repetitions. Also used during a kinetic scan

    Version 2 attributes:
    --------------------
    poly_coeff : array
        Wavelength calibration polynomial coefficients.

    calib_poly : numpy.poly1d
        calibration polynomial


    Version 3 attributes:
    ---------------------
    grating : str
        Description of the used grating

    exposureTime : int
        The exposure time in ms

    TempSet : int
        The set temperature of the camera

    TempRead : int
        The read temperature of the camera

    roi : dict
        Region of interest 
        

    Examples
    --------
    sp = PrincetonSPEFile3("blabla.spe")
    """
    dataTypeDict = {
        0 : ('f', 4, 'float32'),
        1 : ('i', 4, 'int32'),
        2 : ('h', 2, 'int16'),
        3 : ('H', 2, 'int32'),
        5 : ('d', 8, 'float64'),
        6 : ('B', 1, 'int8'),
        8 : ('I', 4, 'int32'),
    }
    def __init__(self, fname, verbose=False):
        self._verbose = verbose
        
        # if not os.path.isfile(fname) and \
        #    not os.path.islink(fname):
        #     raise IOError('%s does not exist' % fname)

        self._spe = open(fname, 'rb')
        self._fname = fname
        self.readData()

    def readData(self):
        """Read all the data into the class."""
        self._readHeader()
        self._readSize()
        self._read_v2_header()
        self._readData()
        self._readFooter()

    def _readFromHeader(self, fmt, offset):
        """Helper function to read bytes from the header."""
        return struct.unpack_from(fmt, self._header, offset)

    def _readHeader(self):
        """Reads the header."""
        # Header length is a fixed number
        nBytesHeader = 4100

        # Read the entire header
        self._header = self._spe.read(nBytesHeader)
        self.headerVersion = self._readFromHeader('f', 1992)[0]
        self._nBytesFooter = self._readFromHeader('I', 678)[0]

    def _readSize(self):
        """ Reads size of the data."""
        self.xdim = self._readFromHeader('H', 42)[0]
        self.ydim = self._readFromHeader('H', 656)[0]
        self.datatype = self._readFromHeader('h', 108)[0]
        self.NumFrames = self._readFromHeader("i", 1446)[0]
        self.xDimDet = self._readFromHeader("H", 6)[0]
        self.yDimDet = self._readFromHeader("H", 18)[0]

    def _read_v2_header(self):
        """Read calibrations parameters and calculate wavelength as it
        was usually done in pre v3 .spe time."""

        # General meta data
        self.exposureTime = timedelta(
            seconds = self._readFromHeader('f', 10)[0] # in seconds
        )
        date = self._readFromHeader('9s', 20)[0].decode('utf-8')
        self.tempSet = self._readFromHeader('f', 36)[0]
        timeLocal = self._readFromHeader('6s', 172)[0].decode('utf-8')
        timeUTC = self._readFromHeader('6s', 179)[0].decode('utf-8')

        # Try statement is a workaround, because sometimes date seems to be male formated
        # and currently the program cant deal with it.
        try:
            self.date = datetime.strptime(
                date + timeLocal, "%d%b%Y%H%M%S"
            )
            self.timeUTC = datetime.strptime(
                date + timeUTC, "%d%b%Y%H%M%S"
            )
        except ValueError:
            if debug:
                print('Malformated date in %s' % self._fname)
                print('date string is: %s' % date)
            self.date = datetime.now()
            self.timeUTC = datetime.now()
        self.gain = self._readFromHeader('I', 198)[0]
        self.comments = self._readFromHeader('400s', 200)[0].decode('utf-8')
        self.central_wl = self._readFromHeader('f', 72)[0] # in nm

        # Lets allways have a wavelength array
        # in worst case its just pixels
        # Read calib data
        self.wavelength = np.arange(self.xdim)
        if self.headerVersion >= 3:
            return
        self.poly_coeff = np.array(self._readFromHeader('6d', 3263))
        # numpy needs polynomparams in reverse oder
        params = self.poly_coeff[np.where( self.poly_coeff != 0 )][::-1]
        if len(params) > 1:
            self.calib_poly = np.poly1d(params)
            self.wavelength = self.calib_poly(np.arange(self.xdim))
        
        

    def _readData(self):
        """Reads the actual data from the binary file.

        Currently this is limited to rectangular data only and
        doesn't support the new fancy data footer features from the
        Version 3 spe file format.
        """
        nPixels = self.xdim * self.ydim

        # fileheader datatypes translated into struct fromatter
        # This tells us what the format of the actual data is
        fmtStr, bytesPerPixel, npfmtStr = self.dataTypeDict[self.datatype]
        fmtStr = str(self.xdim * self.ydim) + fmtStr
        if self._verbose:
            print( "fmtStr = ", fmtStr)

        # Bytes per frame
        nBytesPerFrame = nPixels * bytesPerPixel
        if self._verbose:
            print("nbytesPerFrame = ", nBytesPerFrame)

        # Now read the image data
        # Loop over each image frame in the image
        if self._verbose:
            print('Reading frame number:')

        nBytesHeader = 4100
        self._spe.seek(nBytesHeader)
        self.data = []
        # Todo read until footer here
        #self._data = self._spe.read()
        for ii in range(self.NumFrames):
            data = self._spe.read(nBytesPerFrame)
            if self._verbose:
                print(ii)

            dataArr = np.array(
                struct.unpack_from('='+fmtStr, data, offset=0),
                dtype = npfmtStr
            )
            dataArr.resize((self.ydim, self.xdim))
            self.data.append(dataArr)
        self.data = np.array(self.data)
        return self.data

    def _readFooter(self):
        """ Reads the xml footer from the Version 3 spe file """
        import xmltodict

        if self.headerVersion < 3:
            return
        nBytesFooter = self._nBytesFooter
        self._spe.seek(nBytesFooter)
        self._footer = xmltodict.parse(self._spe.read())
        self.wavelength = np.fromstring(
            self._footer["SpeFormat"]["Calibrations"]["WavelengthMapping"]['Wavelength']['#text'],
            sep=","
        )
        self.central_wl = float(
            self._footer["SpeFormat"]["DataHistories"]['DataHistory']["Origin"]["Experiment"]["Devices"]['Spectrometers']["Spectrometer"]["Grating"]["CenterWavelength"]['#text']
            )
        self.grating = self._footer["SpeFormat"]["DataHistories"]['DataHistory']["Origin"]["Experiment"]["Devices"]['Spectrometers']["Spectrometer"]["Grating"]['Selected']['#text']
        # expusure Time in ms
        self.exposureTime = float(self._footer['SpeFormat']['DataHistories']['DataHistory']['Origin']['Experiment']['Devices']['Cameras']['Camera']['ShutterTiming']['ExposureTime']['#text'])
        temp = self._footer['SpeFormat']['DataHistories']['DataHistory']['Origin']['Experiment']['Devices']['Cameras']['Camera']['Sensor']['Temperature']
        self.tempSet = int(temp['SetPoint']['#text'])
        self.tempRead = int(temp['Reading']['#text'])
        self.roi = self._footer['SpeFormat']['DataHistories']['DataHistory']['Origin']['Experiment']['Devices']['Cameras']['Camera']['ReadoutControl']['RegionsOfInterest']['Result']['RegionOfInterest']
