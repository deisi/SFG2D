"""IO Module to import data from the veronica labview programm """

import os
import copy
import warnings
import numpy as np

from sfg2d.utils.consts import PIXEL, SPECS

names = (
    'pixel',
    'spec_0',
    'spec_1',
    'spec_2',
    'ratio_0',
    'ratio_1'
)
debug=0


def pixel_to_nm(x, central_wl):
    """ transform pixel to nanometer

    Parameters
    ----------
    central_wl : int
        central wavelength of the camera in nm"""

    params_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
    )
    params = np.load(params_file_path)
    calib_cw = int(params_file_path[-7:-4])
    pixel_to_nm = np.poly1d(params) + central_wl - calib_cw
    return pixel_to_nm(x)


def nm_to_pixel(x, central_wl):
    """ transform nm to pixel coordinates for central wavelength

    Parameters
    ----------
    x : array like
        nm to transform in pixel
    central_wl : int
        central wavelength of the camera

    Returns
    -------
    num or array of x in pixel coordinates
    """

    params_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/calib/params_Ne_670.npy"
    )
    params = np.load(params_file_path)
    calib_cw = int(params_file_path[-7:-4])
    if len(params) > 2:
        params = params[-2:]
    if len(params) < 2:
        warnings.Warn("Can't use constant calibration")
    nm_to_pixel = lambda x: (x - params[1] - central_wl + calib_cw)/params[0]

    return nm_to_pixel(x)


def get_from_veronika(fpath):
    """Read files saved by veronika labview programm

    The function reads a file from veronika labview,
    and returns it as a unified 4 dimensional numpy array. And an
    array with all the pump probe time delays.

    Parameters
    ----------
    fpath: str
        Path to load data from.

    Returns
    -------
    tuple of two arrays.
    First array is:
        4 Dimensional numpy array with:
            0 index pp_delays,
            1 index number of repetitions
            2 index number of y-pixel/spectra/bins
            3 index x-pixel number
    """
    raw_data = np.genfromtxt(fpath)
    pp_delays = np.array([0])

    # File is just a simple scan
    if raw_data.shape == (PIXEL, 6):
        ret = np.array([[raw_data.T[1:4]]])
        return ret, pp_delays

    # Check that the shape is readable
    if (raw_data.shape[1])%6 != 0:
        raise IOError("Cant read data in %s" % fpath)

    # We delete the first scan block, because it
    # is the average of the others.
    raw_data = raw_data[:, 6:]

    # Delete the all 0 lines, that mark the end of a block
    if debug > 2:
        print("Verify, that only empty values have beed removed:")
        print(raw_data[PIXEL+1::PIXEL+2])
    raw_data = np.delete(raw_data, slice(PIXEL+1,None,PIXEL+2), 0)

    num_rows, num_columns = raw_data.shape

    # Every 1600 lines there is an additional line with the pp_delays
    num_pp_delays = num_rows%(PIXEL)
    # The first colum is only pixel number
    num_repetitions = num_columns//6
    pp_delays = raw_data[::PIXEL+1][:, 0]
    # Delete pp_delay rows
    raw_data = np.delete(raw_data, slice(None, None, PIXEL+1), 0)

    # Delete pixel number columns
    raw_data = np.delete(raw_data, slice(None, None, 6), 1)

    # Delete lines, that are the division or avg of other lines
    raw_data = np.delete(raw_data, slice(None, None, 5), 1)
    raw_data = np.delete(raw_data, slice(3, None, 4), 1)

    # Init container for the result.
    ret = np.zeros((num_pp_delays, num_repetitions, SPECS, PIXEL), dtype=raw_data.dtype)
    for rep_index in range(num_repetitions):
        for pp_delay_index in range(num_pp_delays):
            column_slice = slice(PIXEL*pp_delay_index, PIXEL*pp_delay_index + PIXEL)
            row_slice = slice(rep_index*SPECS, rep_index*SPECS+SPECS)
            ret[pp_delay_index, rep_index] = raw_data[column_slice, row_slice].T

    ret = ret.astype('long')
    return ret, pp_delays




# all below this point is DEPRECATED

def read_save(fpath, **kwargs):
    """read files saved with veronica save button """

    ret = read_csv(
        fpath,
        sep='\t',
        header=None,
        names=names,
        usecols=[0, 1, 2, 3],
        **kwargs
    )

    # metadata based on filename makes the calibration
    metadata = get_metadata_from_filename(fpath)
    ret = ret.astype('uint32')

    if metadata["central_wl"] == -1:
        ret.set_index("pixel", inplace=True)
    else:
        pixel = np.arange(PIXEL)
        nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
        wavenumber = np.round(nm_to_ir_wavenumbers(
            nm, up_wl=metadata.get('vis_wl', 800)
        ), decimals=1)
        ret.drop("pixel", inplace=True, axis=1)
        ret.set_index(wavenumber, inplace=True)
        ret.index.name = 'wavenumber'
        ret.sort_index(inplace=True)

    ret = Scan(ret, metadata=copy.deepcopy(metadata))

    return ret


def read_scan_stack(fpath, **kwargs):
    """import veronica scan files containing only repetition """
    ret = read_csv(
        fpath,
        sep='\t',
        header=None,
        skiprows=1,
        skipfooter=1,
        engine="python",
        **kwargs
    )

    # Check for correct data shape
    if ret.shape[1] % 6 is not 0:
        raise IOError("Cant understand shape of data in %s" % fpath)

    # number of spectrum repetition
    reps = ret.shape[1]//6

    def _iterator(elm):
        return np.array(
            # ditch first repetition because it is the AVG
            [elm(i) for i in range(1, reps)]
        ).flatten().tolist()

    # Keep only interesting columns
    colum_inds = [0] + _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
    ret = ret[ret.columns[colum_inds]]

    # Set colum names
    names = ['pixel'] + _iterator(lambda i: ['spec_0', 'spec_1', 'spec_2'])
    ret.columns = names

    # metadata based on filename makes the calibration
    metadata = get_metadata_from_filename(fpath)

    # Data types are integers
    ret = ret.astype('uint32')

    # Cecause central wavelength migh be missing
    if metadata["central_wl"] == -1:
        # Use pixel as index
        ret.set_index("pixel", inplace=True)
    else:
        pixel = np.arange(PIXEL)
        nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
        wavenumber = np.round(nm_to_ir_wavenumbers(
            nm, up_wl=metadata.get('vis_wl', 800)
        ), decimals=1)
        ret.drop('pixel', axis=1, inplace=True)
        ret.set_index(wavenumber, inplace=True)
        ret.index.name = 'wavenumber'
        ret.sort_index(inplace=True)

    return Scan(ret, metadata=copy.deepcopy(metadata))


def read_time_scan(fpath, **kwargs):
    """ """
    ret = read_csv(
        fpath,
        sep='\t',
        header=None,
        **kwargs
    )

    # Check for correct data shape
    if ret.shape[1] % 6 is not 0 and ret.shape[1] % 1602 is not 0:
        raise IOError("Cant understand shape of data in %s" % fpath)

    # number of spectrum repetition
    reps = ret.shape[1]//6

    def _iterator(elm):
        return np.array(
            # ditch first iteration because it the AVG
            [elm(i) for i in range(1, reps)]
        ).flatten().tolist()

    # Remove uninteresting columns
    colum_inds = [0] + _iterator(lambda i: [6*i+2, 6*i+3, 6*i+4])
    ret = ret[ret.columns[colum_inds]]

    # time delays so we can use them for multiaxes
    pp_delays = ret.iloc[0::1602, 0]

    # Data type is integer
    ret = ret.astype('uint32')

    # Set colum names
    names = ['pixel'] + _iterator(lambda i: ['spec_0', 'spec_1', 'spec_2'])
    ret.columns = names

    # drop lines with pp_delays
    ret.drop(pp_delays.index, inplace=True)

    # remove empty lines like the 1600 row
    ret = ret[~np.all(ret == ret.iloc[1600], 1)]

    # drop pixel column. If needed its added later again
    ret.drop('pixel', axis=1, inplace=True)

    # metadata based on filename makes the calibration and are used to
    # index the dataframe
    metadata = get_metadata_from_filename(fpath)
    pixel = np.arange(PIXEL)

    if metadata["central_wl"] == -1:
        # Use pixel as index
        pp_delays = pp_delays.as_matrix()
        index = MultiIndex.from_product(
            (pp_delays, pixel),
            names=('pp_delay', 'pixel')
        )
    else:
        nm = pixel_to_nm(pixel, central_wl=metadata['central_wl'])
        wavenumber = np.round(nm_to_ir_wavenumbers(
            nm, up_wl=metadata.get('vis_wl', 800)
        ), decimals=1)

        # make indeces
        pp_delays = pp_delays.as_matrix()
        index = MultiIndex.from_product(
            (pp_delays, wavenumber),
            names=('pp_delay', 'wavenumber')
        )
    ret.index = index
    ret.sort_index(inplace=True)

    # Link Repeated Columns together
    return TimeScan(ret, metadata=copy.deepcopy(metadata))


def read_auto(fpath, **kwargs):
    """ use fpath and datashape to automatically determine data type
    of fpath and use according read function to import data."""
    folder, ffile = os.path.split(fpath)
    metadata = get_metadata_from_filename(fpath)

    def _guess_sepctrum_type_by_content(fpath):
        warnings.warn('cant determine spectrum type of data by filename.'
                      'Trying to determine datatype from content.'
                      'This is much slower.')
        ret = read_csv(
            fpath,
            sep='\t',
            header=None,
        )
        if ret.shape == (1600, 6):
            metadata['sp_type'] = 'sp'
        elif ret.shape[0] == 1602 and ret.shape[1] % 6 is 0:
            metadata['sp_type'] = 'sc'
        elif ret.shape[0] % 1602 is 0 and ret.shape[1] % 6 is 0:
            metadata['sp_type'] = 'ts'
        else:
            raise IOError(
                "Can't determine spectrum type of %s\nshape is %s" %
                (fpath, ret.shape)
            )
        return ret

    # check if name determines spectrum type
    if metadata['sp_type'] == 'sp' or \
       metadata['sp_type'] == 'sc' or \
       metadata['sp_type'] == 'ts':
        pass
    else:
        ret = _guess_sepctrum_type_by_content(fpath)

    # If name wa entered wringly, this can still be wrong
    try:
        if metadata['sp_type'] == 'sp':
            ret = read_save(fpath, **kwargs)

        elif metadata['sp_type'] == 'sc':
            ret = read_scan_stack(fpath, **kwargs)

        elif metadata['sp_type'] == 'ts':
            ret = read_time_scan(fpath, **kwargs)

        # Needed in the case of content import. Else no harm
        ret.metadata['sp_type'] = copy.deepcopy(metadata['sp_type'])
    # happens if read function was chosen by wrong or iritating filename.
    # Try to read data one more time using actual shape. If this does not
    # work as well we give up.
    except ValueError:
        warnings.warn(
            "Wrong metadata for %s. Type %s was given" %
            (fpath, metadata['sp_type'])
        )
        ret = _guess_sepctrum_type_by_content(fpath)
        if metadata['sp_type'] == 'sp':
            ret = read_save(fpath, **kwargs)

        elif metadata['sp_type'] == 'sc':
            ret = read_scan_stack(fpath, **kwargs)

        elif metadata['sp_type'] == 'ts':
            ret = read_time_scan(fpath, **kwargs)

        else:
            raise IOError("Cant understand datatype of %s" % fpath)

    return ret


def get_scan(fpath, fbase=None, fnorm=None, sub_base=True, div_norm=True):
    ret = read_auto(fpath)

    if fbase:
        ret.base = read_auto(fbase).med
        if sub_base:
            ret.sub_base(inplace=True)

    if fnorm:
        ret.norm = read_auto(fnorm).med
        if div_norm:
            ret.normalize(inplace=True)

    return ret


def get_time_scan(fpath, base=None, norm=None):
    ret = read_time_scan(fpath)
    ret.df.drop("pixel", axis=1, inplace=True)
    if getattr(base):
        ret.base = base
        ret.sub_base(inplace=True)

    if getattr(norm):
        ret.norm = norm
        ret.normalize(inplace=True)
