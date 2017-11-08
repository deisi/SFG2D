import os, yaml


X_PIXEL_INDEX = -1 #  Axis of x-pixel data
Y_PIXEL_INDEX = -2 # Axis of y-pixel data
SPEC_INDEX = -2 # Axis of spectra
FRAME_AXIS_INDEX = -3 # Axis of frames
PP_INDEX = -4 # Axis of pp_delays
X_PIXEL_INDEX_P = 3

SPEC_INDEX_P = 2
FRAME_AXIS_INDEX_P = 1
PP_INDEX_P = 0
PIXEL = 1600 # Number of x-pixel
SPECS = 3 # Number of binned spectra
VIS_WL = None # Default Wavelength of the visible
PUMP_FREQ = None # Default pump frequency
NORM_SPEC = None # Default 4d spectrum for normalization
BASE_SPEC = None # Default 4d Baseline/Background spectrum
STEPSIZE = 0.1 # Smallest stepsize used for decay function
XE = None # Equidistant sampling distance for decay function
XG = None # Equidistant sampling for gaussian instrumense response function

#Key is propertie names of SfgRecord properties and Value is default axis labels.
x_property2_label = {
    'wavenumber': 'Wavenumber in 1/cm',
    'wavelength': 'Wavelength in nm',
    'pixel': 'Pixel Number',
    'pp_delays': 'Time in fs',
    'pp_delays_ps': 'Time in ps',
}


### Read and apply user config
try:
    with open(os.path.expanduser("~/.sfg2d.yaml"), 'r') as conf:
        USER_CONFIG = yaml.load(conf)
except FileNotFoundError:
    USER_CONFIG = None

if not USER_CONFIG:
    USER_CONFIG = {}
    USER_CONFIG['calibration'] = {}
    USER_CONFIG['calibration'].setdefault('file', None)
    USER_CONFIG['calibration'].setdefault('calib_cw', 670)

if not USER_CONFIG.get('calibration'):
    USER_CONFIG['calibration'] = {}
    USER_CONFIG['calibration'].setdefault('file', None)
    USER_CONFIG['calibration'].setdefault('calib_cw', 670)

if not USER_CONFIG['calibration'].get('file'):
    USER_CONFIG['calibration'].setdefault('file', None)

if not USER_CONFIG['calibration'].get('calib_cw'):
    USER_CONFIG['calibration'].setdefault('calib_cw', 670)

print(USER_CONFIG)

CALIB_PARAMS_FILEPATH = USER_CONFIG['calibration']['file']
if not CALIB_PARAMS_FILEPATH:
    CALIB_PARAMS_FILEPATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
       )

if os.path.splitext(CALIB_PARAMS_FILEPATH)[-1] == '.npy':
    from numpy import load
    CALIB_PARAMS = load(CALIB_PARAMS_FILEPATH)
else:
    from numpy import loadtxt
    CALIB_PARAMS = loadtxt(CALIB_PARAMS_FILEPATH)

CALIB_CW = int(USER_CONFIG['calibration']['calib_cw'])
