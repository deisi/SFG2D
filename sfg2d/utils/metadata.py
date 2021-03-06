import os
import re
from datetime import timedelta, datetime

debug = 0

def get_metadata_from_filename(fpath):
    """Function to extract metadata from filenames.

    Returns
    -------
    Dictionary with metadata information."""
    fpath = os.path.normpath(fpath)
    metadata = {}
    ffolder, ffile = os.path.split(fpath)
    fsplit = ffile.split("_")
    metadata["uri"] = os.path.abspath(fpath)
    metadata['filename'] = os.path.splitext(ffile)[0]
    try:
        ext = os.path.splitext(fpath)[1]
        if ext == '.dat':
            metadata["sp_type"] = fsplit[1]
            metadata["material"] = fsplit[2]
        elif ext == '.spe':
            metadata["material"] = fsplit[1]
    except IndexError:
        pass

    central_wl_conditions = [
        '_wl(\d\d\d)_',
        '_w(\d\d\d)_',
        '_cw(\d\d\d)_',
    ]
    for condition in central_wl_conditions:
        match_central_wl = re.search(condition, ffile)
        if match_central_wl:
            metadata['central_wl'] = int(match_central_wl.group(1))
            break

    # find gain
    match_gain = re.search("_g([cm\d]+)_", ffile)
    if match_gain:
        if match_gain.group(1) == "cm":
            metadata["gain"] = -1
        else:
            metadata["gain"] = int(match_gain.group(1))

    # find exposiure time
    match_exp_time = re.search("_e(\d+)([msh]+)_", ffile)
    if match_exp_time:
        time = match_exp_time.group(1)
        unit = match_exp_time.group(2)
        timedelta_kwargs = {}

        # Translate between timedelta and my naming convention.
        if unit == 'm':
            timedelta_kwargs = {"minutes" : int(time)}
        elif unit == 's':
            timedelta_kwargs = {"seconds" : int(time)}
        elif unit == 'ms':
            timedelta_kwargs = {"milliseconds" : int(time)}
        elif unit == 'h':
            timedelta_kwargs = {"hours" : int(time)}
        if debug:
            print(time, unit)
        exp_time = timedelta(**timedelta_kwargs)
        metadata["exposure_time"] = exp_time

    # find polarisation
    if "_ppp_" in ffile:
        metadata["polarisation"] = "ppp"
    elif "_ssp_" in ffile:
        metadata["polarisation"] = "ssp"

    # find rotation
    match  = re.search('_rot(\d+)', ffile)
    if match:
        metadata['rot'] = int(match.group(1))

    # pump status
    def _match_booleans(denom, key):
        pattern = "_" + denom + "([01])_"
        match = re.search(pattern, ffile)
        if match:
            metadata[key] = bool(match.group(1))

    # get creation time of the file
    try:
        date = datetime.fromtimestamp(os.path.getctime(fpath))
        metadata['date'] = date
    except FileNotFoundError:
        pass

    _match_booleans("pu", "pump")
    _match_booleans("pr", "probe")
    _match_booleans("vis", "vis")
    _match_booleans("gal", "galvaner")
    _match_booleans("chop", "chopper")
    _match_booleans("purge", "purge")

    # Get temperature of sample
    match_temp = re.search("_(\d+)C", ffile)
    match_temp2 = re.search("_C(\d+)", ffile)
    if match_temp:
        metadata["Temperature"] = match_temp.group(1)
    elif match_temp2:
        metadata["Temperature"] = match_temp2.group(1)
    return metadata


def time_scan_time(pp_delays, exp_time, reps):
    """Return time, when the measurment is done.

    Parameters
    ----------
    pp_delays : int
        Number of different pump-probe time delays that are scaned.

    exp_time : int
        Number of seconds of a individual scan.

    reps : int
        Number of repetitions to be done

    Returns
    -------
    datetime obj showing when the scan is done."""
    import datetime
    ret = datetime.datetime.now()
    ret += datetime.timedelta(minutes=pp_delays*exp_time*reps)
    return ret


def get_unit_from_string(string):
    """Return [] encapsulated unit from string."""
    match = re.search('\[(.*)\]', string)
    ret = ''
    if match:
        ret = match.group(1)
    return ret
