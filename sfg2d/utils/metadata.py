import os
import re
import copy
from datetime import timedelta, datetime

debug = 0

# A skeleton for a Metadata dictionary
MetaData = {
    "uri": "",
    "sp_type": "",
    "material": "",
    "central_wl": -1,
    "vis_wl": 810,
    "gain": -2,
    "exposure_time": timedelta(0),
    "polarisation": "",
    "pump": None,
    "probe": None,
    "vis": None,
    "galvaner": None,
    "chopper": None,
    "purge": None,
    "date": None,
    "pump_freq": 0, # frequency of the pump in 1/cm
    "pump_width": 0, # width of the pump in 1/cm
}


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
    try:
        ext = os.path.splitext(fpath)[1]
        if ext == '.dat':
            metadata["sp_type"] = fsplit[1]
            metadata["material"] = fsplit[2]
        elif ext == '.spe':
            metadata["material"] = fsplit[1]
    except IndexError:
        pass

    #  try:
    #      metadata["central_wl"] = int(fsplit[3].split("w")[1])
    #  except IndexError:
    #      match_cw = re.search("_wl(\d+)", ffile)
    #      if match_cw:
    #          metadata["central_wl"] = int(match_cw.group(1))
    #  except ValueError:
    #      pass
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
