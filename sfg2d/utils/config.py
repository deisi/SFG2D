import os, yaml, logging

CONFIG = dict(
    X_PIXEL_INDEX = -1, #  Axis of x-pixel data
    Y_PIXEL_INDEX = -2, # Axis of y-pixel data
    SPEC_INDEX = -2, # Axis of spectra
    FRAME_AXIS_INDEX = -3, # Axis of frames
    PP_INDEX = -4, # Axis of pp_delays
    X_PIXEL_INDEX_P = 3,
    SPEC_INDEX_P = 2,
    FRAME_AXIS_INDEX_P = 1,
    PP_INDEX_P = 0,
    PIXEL = 1600, # Number of x-pixel
    SPECS = 3, # Number of binned spectra
    VIS_WL = None, # Default Wavelength of the visible
    PUMP_FREQ = None, # Default pump frequency
    NORM_SPEC = None, # Default 4d spectrum for normalization
    BASE_SPEC = None, # Default 4d Baseline/Background spectrum
    STEPSIZE = 0.1, # Smallest stepsize used for decay function
    XE = None, # Equidistant sampling distance for decay function
    XG = None, # Equidistant sampling for gaussian instrumense response function
)

### Read and apply user config
try:
    with open(os.path.expanduser("~/.sfg2d.yaml"), 'r') as conf:
        USER_CONFIG = yaml.load(conf)
        if not USER_CONFIG:
            USER_CONFIG = {}
        # We want user config to overwrite default config
        CONFIG = {**CONFIG, **USER_CONFIG}
        logging.info("Loading user config.")
except FileNotFoundError:
    logging.info("No user config found.")


CONFIG.setdefault('calibration', {})
CONFIG['calibration'].setdefault('file', None)
CONFIG['calibration'].setdefault('calib_cw', 670)

logging.info("User Config: {}".format(CONFIG))

CALIB_PARAMS_FILEPATH = CONFIG['calibration']['file']
#if CALIB_PARAMS_FILEPATH:
if not CALIB_PARAMS_FILEPATH:
    CONFIG['CALIB_PARAMS_FILEPATH'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
       )
    CALIB_PARAMS_FILEPATH = CONFIG['CALIB_PARAMS_FILEPATH']

if os.path.splitext(CALIB_PARAMS_FILEPATH)[-1] == '.npy':
    from numpy import load
    CONFIG['CALIB_PARAMS'] = load(CALIB_PARAMS_FILEPATH)
else:
    from numpy import loadtxt
    CONFIG['CALIB_PARAMS'] = loadtxt(CALIB_PARAMS_FILEPATH)

CONFIG['CALIB_CW'] = int(CONFIG['calibration']['calib_cw'])
