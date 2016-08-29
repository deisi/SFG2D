import os
import re

MetaData = {
    "uri" : "",
    "sp_type" : "",
    "material" : "",
    "central_wl" : -1,
    "vis_wl" : 810,
    "gain" : -2,
    "exposure_time" : (), # exposure time is (value, unit) tuple
    "polarisation" : "",
    "pump" : None,
    "probe" : None,
    "vis" : None,
    "galvaner" : None,
    "chopper" : None,
    "purge" : None
}

def get_metadata_from_filename(fpath):
    fpath = os.path.normpath(fpath)
    metadata = MetaData
    ffolder, ffile = os.path.split(fpath)
    fsplit = ffile.split("_")
    metadata["uri"] = os.path.abspath(fpath)
    metadata["sp_type"] = fsplit[1]
    metadata["material"] = fsplit[2]
    try:
        metadata["central_wl"] = int(fsplit[3].split("w")[1])
    except IndexError:
        match_cw = re.search("_wl(\d+)", ffile)
        if match_cw:
            metadata["central_wl"] = int(match_cw.group(1))

    # find gain
    match_gain = re.search("_g([cm\d]+)_", ffile)
    if match_gain:
        if match_gain.group(1) == "cm":
            metadata["gain"] = -1
        else:
            metadata["gain"] = int(match_gain(1))

    # find exposiure time
    match_exp_time = re.search("_e(\d.)([msh]+)_", ffile)
    if match_exp_time:
        metadata["exposure_time"] = int(match_exp_time.group(1)), match_exp_time.group(2)
    
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

    _match_booleans("pu", "pump")
    _match_booleans("pr", "probe")
    _match_booleans("vis", "vis")
    _match_booleans("gal", "galvaner")
    _match_booleans("chop", "chopper")
    _match_booleans("purge", "purge")
    
    return metadata
