import numpy
import struct

def read_spe(spefilename, verbose=False):
    """ 
    Read a binary PI SPE file into a python dictionary

    Inputs:

        spefilename --  string specifying the name of the SPE file to be read
        verbose     --  boolean print debug statements (True) or not (False)

        Outputs
        spedict     
        
            python dictionary containing header and data information
            from the SPE file
            Content of the dictionary is:
            spedict = {'data':[],    # a list of 2D numpy arrays, one per image
            'IGAIN':pimaxGain,
            'EXPOSURE':exp_sec,
            'SPEFNAME':spefilename,
            'OBSDATE':date,
            'CHIPTEMP':detectorTemperature
            }

    I use the struct module to unpack the binary SPE data.
    Some useful formats for struct.unpack_from() include:
    fmt   c type          python
    c     char            string of length 1
    s     char[]          string (Ns is a string N characters long)
    h     short           integer 
    H     unsigned short  integer
    l     long            integer
    f     float           float
    d     double          float

    The SPE file defines new c types including:
        BYTE  = unsigned char
        WORD  = unsigned short
        DWORD = unsigned long


    Example usage:
    Given an SPE file named test.SPE, you can read the SPE data into
    a python dictionary named spedict with the following:
    >>> import piUtils
    >>> spedict = piUtils.readSpe('test.SPE')
    """
  
    # open SPE file as binary input
    spe = open(spefilename, "rb")
    
    # Header length is a fixed number
    nBytesInHeader = 4100

    # Read the entire header
    header = spe.read(nBytesInHeader)
    
    # version of WinView used
    swversion = struct.unpack_from("16s", header, offset=688)[0]
    
    # version of header used
    # Eventually, need to adjust the header unpacking
    # based on the headerVersion.  
    headerVersion = struct.unpack_from("f", header, offset=1992)[0]
  
    # which camera controller was used?
    controllerVersion = struct.unpack_from("h", header, offset=0)[0]
    if verbose:
        print("swversion         = ", swversion)
        print("headerVersion     = ", headerVersion)
        print("controllerVersion = ", controllerVersion)
    
    # Date of the observation
    # (format is DDMONYYYY  e.g. 27Jan2009)
    date = struct.unpack_from("9s", header, offset=20)[0]
    
    # Exposure time (float)
    exp_sec = struct.unpack_from("f", header, offset=10)[0]
    
    # Intensifier gain
    pimaxGain = struct.unpack_from("h", header, offset=148)[0]

    # Not sure which "gain" this is
    gain = struct.unpack_from("H", header, offset=198)[0]
    
    # Data type (0=float, 1=long integer, 2=integer, 3=unsigned int)
    data_type = struct.unpack_from("h", header, offset=108)[0]

    comments = struct.unpack_from("400s", header, offset=200)[0]

    # CCD Chip Temperature (Degrees C)
    detectorTemperature = struct.unpack_from("f", header, offset=36)[0]

    # The following get read but are not used
    # (this part is only lightly tested...)
    analogGain = struct.unpack_from("h", header, offset=4092)[0]
    noscan = struct.unpack_from("h", header, offset=34)[0]
    pimaxUsed = struct.unpack_from("h", header, offset=144)[0]
    pimaxMode = struct.unpack_from("h", header, offset=146)[0]

    ########### here's from Kasey
    #int avgexp 2 number of accumulations per scan (why don't they call this "accumulations"?)
#TODO: this isn't actually accumulations, so fix it...    
    accumulations = struct.unpack_from("h", header, offset=668)[0]
    if accumulations == -1:
        # if > 32767, set to -1 and 
        # see lavgexp below (668) 
        #accumulations = struct.unpack_from("l", header, offset=668)[0]
        # or should it be DWORD, NumExpAccums (1422): Number of Time experiment accumulated        
        accumulations = struct.unpack_from("l", header, offset=1422)[0]
        
    """Start of X Calibration Structure (although I added things to it that I thought were relevant,
       like the center wavelength..."""
    xcalib = {}
    
    #SHORT SpecAutoSpectroMode 70 T/F Spectrograph Used
    xcalib['SpecAutoSpectroMode'] = bool( struct.unpack_from("h", header, offset=70)[0] )

    #float SpecCenterWlNm # 72 Center Wavelength in Nm
    xcalib['SpecCenterWlNm'] = struct.unpack_from("f", header, offset=72)[0]
    
    #SHORT SpecGlueFlag 76 T/F File is Glued
    xcalib['SpecGlueFlag'] = bool( struct.unpack_from("h", header, offset=76)[0] )

    #float SpecGlueStartWlNm 78 Starting Wavelength in Nm
    xcalib['SpecGlueStartWlNm'] = struct.unpack_from("f", header, offset=78)[0]

    #float SpecGlueEndWlNm 82 Starting Wavelength in Nm
    xcalib['SpecGlueEndWlNm'] = struct.unpack_from("f", header, offset=82)[0]

    #float SpecGlueMinOvrlpNm 86 Minimum Overlap in Nm
    xcalib['SpecGlueMinOvrlpNm'] = struct.unpack_from("f", header, offset=86)[0]

    #float SpecGlueFinalResNm 90 Final Resolution in Nm
    xcalib['SpecGlueFinalResNm'] = struct.unpack_from("f", header, offset=90)[0]

    #  short   BackGrndApplied              150  1 if background subtraction done
    xcalib['BackgroundApplied'] = struct.unpack_from("h", header, offset=150)[0]
    BackgroundApplied=False
    if xcalib['BackgroundApplied']==1: BackgroundApplied=True

    #  float   SpecGrooves                  650  Spectrograph Grating Grooves
    xcalib['SpecGrooves'] = struct.unpack_from("f", header, offset=650)[0]

    #  short   flatFieldApplied             706  1 if flat field was applied.
    xcalib['flatFieldApplied'] = struct.unpack_from("h", header, offset=706)[0]
    flatFieldApplied=False
    if xcalib['flatFieldApplied']==1: flatFieldApplied=True
    
    #double offset # 3000 offset for absolute data scaling */
    xcalib['offset'] = struct.unpack_from("d", header, offset=3000)[0]

    #double factor # 3008 factor for absolute data scaling */
    xcalib['factor'] = struct.unpack_from("d", header, offset=3008)[0]
    
    #char current_unit # 3016 selected scaling unit */
    xcalib['current_unit'] = struct.unpack_from("c", header, offset=3016)[0]

    #char reserved1 # 3017 reserved */
    xcalib['reserved1'] = struct.unpack_from("c", header, offset=3017)[0]

    #char string[40] # 3018 special string for scaling */
    xcalib['string'] = struct.unpack_from("40c", header, offset=3018)
    
    #char reserved2[40] # 3058 reserved */
    xcalib['reserved2'] = struct.unpack_from("40c", header, offset=3058)

    #char calib_valid # 3098 flag if calibration is valid */
    xcalib['calib_valid'] = struct.unpack_from("c", header, offset=3098)[0]

    #char input_unit # 3099 current input units for */
    xcalib['input_unit'] = struct.unpack_from("c", header, offset=3099)[0]
    """/* "calib_value" */"""

    #char polynom_unit # 3100 linear UNIT and used */
    xcalib['polynom_unit'] = struct.unpack_from("c", header, offset=3100)[0]
    """/* in the "polynom_coeff" */"""

    #char polynom_order # 3101 ORDER of calibration POLYNOM */
    xcalib['polynom_order'] = struct.unpack_from("c", header, offset=3101)[0]

    #char calib_count # 3102 valid calibration data pairs */
    xcalib['calib_count'] = struct.unpack_from("c", header, offset=3102)[0]

    #double pixel_position[10];/* 3103 pixel pos. of calibration data */
    xcalib['pixel_position'] = struct.unpack_from("10d", header, offset=3103)

    #double calib_value[10] # 3183 calibration VALUE at above pos */
    xcalib['calib_value'] = struct.unpack_from("10d", header, offset=3183)

    #double polynom_coeff[6] # 3263 polynom COEFFICIENTS */
    xcalib['polynom_coeff'] = struct.unpack_from("6d", header, offset=3263)

    #double laser_position # 3311 laser wavenumber for relativ WN */
    xcalib['laser_position'] = struct.unpack_from("d", header, offset=3311)[0]

    #char reserved3 # 3319 reserved */
    xcalib['reserved3'] = struct.unpack_from("c", header, offset=3319)[0]

    #unsigned char new_calib_flag # 3320 If set to 200, valid label below */
    #xcalib['calib_value'] = struct.unpack_from("BYTE", header, offset=3320)[0] # how to do this?

    #char calib_label[81] # 3321 Calibration label (NULL term'd) */
    xcalib['calib_label'] = struct.unpack_from("81c", header, offset=3321)

    #char expansion[87] # 3402 Calibration Expansion area */
    xcalib['expansion'] = struct.unpack_from("87c", header, offset=3402)
    ########### end of Kasey's addition

    if verbose:
        print( "date      = ["+date+"]")
        print( "exp_sec   = ", exp_sec)
        print( "pimaxGain = ", pimaxGain)
        print( "gain (?)  = ", gain)
        print( "data_type = ", data_type)
        print( "comments  = ["+comments+"]")
        print( "analogGain = ", analogGain)
        print( "noscan = ", noscan)
        print( "detectorTemperature [C] = ", detectorTemperature)
        print( "pimaxUsed = ", pimaxUsed)

    # Determine the data type format string for
    # upcoming struct.unpack_from() calls
    if data_type == 0:
        # float (4 bytes)
        dataTypeStr = "f"  #untested
        bytesPerPixel = 4
        dtype = "float32"
    elif data_type == 1:
        # long (4 bytes)
        dataTypeStr = "l"  #untested
        bytesPerPixel = 4
        dtype = "int32"
    elif data_type == 2:
        # short (2 bytes)
        dataTypeStr = "h"  #untested
        bytesPerPixel = 2
        dtype = "int32"
    elif data_type == 3:  
        # unsigned short (2 bytes)
        dataTypeStr = "H"  # 16 bits in python on intel mac
        bytesPerPixel = 2
        dtype = "int32"  # for numpy.array().
        # other options include:
        # IntN, UintN, where N = 8,16,32 or 64
        # and Float32, Float64, Complex64, Complex128
        # but need to verify that pyfits._ImageBaseHDU.ImgCode cna handle it
        # right now, ImgCode must be float32, float64, int16, int32, int64 or uint8
    else:
        print("unknown data type")
        print("returning...")
        sys.exit()
  
    # Number of pixels on x-axis and y-axis
    nx = struct.unpack_from("H", header, offset=42)[0]
    ny = struct.unpack_from("H", header, offset=656)[0]
    
    # Number of image frames in this SPE file
    nframes = struct.unpack_from("l", header, offset=1446)[0]

    if verbose:
        print("nx, ny, nframes = ", nx, ", ", ny, ", ", nframes)
    
    npixels = nx*ny
    npixStr = str(npixels)
    fmtStr  = npixStr+dataTypeStr
    if verbose:
        print("fmtStr = ", fmtStr)
    
    # How many bytes per image?
    nbytesPerFrame = npixels*bytesPerPixel
    if verbose:
        print("nbytesPerFrame = ", nbytesPerFrame)

    # Create a dictionary that holds some header information
    # and contains a placeholder for the image data
    spedict = {'data':[],    # can have more than one image frame per SPE file
                'IGAIN':pimaxGain,
                'EXPOSURE':exp_sec,
                'SPEFNAME':spefilename,
                'OBSDATE':date,
                'CHIPTEMP':detectorTemperature,
                'COMMENTS':comments,
                'XCALIB':xcalib,
                'ACCUMULATIONS':accumulations,
                'FLATFIELD':flatFieldApplied,
                'BACKGROUND':BackgroundApplied
                }
    
    # Now read in the image data
    # Loop over each image frame in the image
    if verbose:
        print("Reading image frames number ",)
    for ii in range(nframes):
        iistr = str(ii)
        data = spe.read(nbytesPerFrame)
        if verbose:
            print (ii," ",)
    
        # read pixel values into a 1-D numpy array. the "=" forces it to use
        # standard python datatype size (4bytes for 'l') rather than native
        # (which on 64bit is 8bytes for 'l', for example).
        # See http://docs.python.org/library/struct.html
        dataArr = numpy.array(struct.unpack_from("="+fmtStr, data, offset=0),
                            dtype=dtype)

        # Resize array to nx by ny pixels
        # notice order... (y,x)
        dataArr.resize((ny, nx))
        #print dataArr.shape

        # Push this image frame data onto the end of the list of images
        # but first cast the datatype to float (if it's not already)
        # this isn't necessary, but shouldn't hurt and could save me
        # from doing integer math when i really meant floating-point...
        spedict['data'].append( dataArr.astype(float) )

    if verbose:
        print ("")
  
    return spedict
