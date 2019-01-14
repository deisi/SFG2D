"""Module to read raw data and produce records with."""
from pylab import *
import sys
import os
import numpy as np
import sfg2d.core as core
import sfg2d.fig as fig
import sfg2d.models
import dpath.util
import logging
from copy import deepcopy
from pip._internal.utils.misc import get_installed_distributions
from . import myyaml as yaml
import pandas as pd
from glob import glob
plt.ion()
logger = logging.getLogger(__name__)

### Constants for configurations and options
MODEL = 'model'
MODELS = 'models'
OPTIONS = 'options'
RECORDS = 'records'
FIGURES = 'figures'
GITVERSION = '.gitversion'
INSTALLED_PACKAGES = '.installed_packages'
CONFIG_FILE = './raw_config.yaml'
MODEL_FILE = './models.yaml'
FILE_CALIB = ('file_calib', None)
MPLSTYLE = ('mplstyle', '/home/malte/sfg2d/styles/presentation.mplstyle')
CACHE_DIR = ('cache_dir', './cache')
FIGURE_FILE = ('figure_file', './figures.pdf')
VIS_WL = ('vis_wl', None)


def get_pd_fitargs(analyser):
    """Return fitargs as DataFrame with model names."""
    search_res = list(dpath.util.search(
        analyser.configuration['models'],
        '*/kwargs_model/fitarg',
        yielded=True
    ))
    model_names = [path.split('/')[0] for path, _ in search_res]
    datas = [fitarg for _, fitarg in search_res]
    record_names = [analyser.configuration['models'][model_name]['record'] for model_name in model_names]
    roi = [analyser.configuration['models'][model_name]['kwargs_select_yerr']['roi_wavenumber'] for model_name in model_names]

    df = pd.DataFrame.from_dict(datas)
    df['model_name'] = model_names
    df['record_name'] = record_names
    df['roi'] = roi
    return df


def read_yaml(fpath):
    with open(fpath) as ifile:
        configuration = yaml.load(ifile)
    return configuration

def save_yaml(fpath, configuration):
    """Save configuration dict to fpath."""
    logger.info(
        'Saving configuration to {}'.format(
            os.path.abspath(fpath)
        )
   )
    with open(fpath, 'w') as ofile:
        configuration = yaml.dump(
            configuration, ofile, default_flow_style=False
        )
    return configuration

def files_to_records(list_of_files, select_names=False, split='_',
                     kwargs_record=None):
    """Import all files as records and return records dict.

    Auxilary function to batch import a list if files.
    **kwargs**:
      - **select_names**: None or slice. The `_` splited part of filename
          to use for record names. User must make shure it is unique. Else
          later imports overwirte older ones.
      - **split**: str, of what to split filename with in case of select_names
      - **kwargs_record**: kwargs passed to the import of eachr record

    **Returns:**
    Dictionary of records, where filenames were used to make up dict keys.
    per default full filename is used for key. If `select_names` slice is given,
    the filename is trimed town to the selected range using `split` as split
    """
    records = {}
    if not kwargs_record:
        kwargs_record = {}

    for fpath in list_of_files:
        logger.debug('Reading {}'.format(fpath))
        logger.debug('kwargs_record {}'.format(kwargs_record))
        record = core.SfgRecord(fpath, **kwargs_record)
        name = os.path.splitext(
            os.path.basename(record.metadata['uri'])
        )[0]
        if select_names:
            name = '_'.join(name.split(split)[select_names])

        records[name] = record
    return records

def metadata_df(records, ignore=None):
    """Make a pandas data frame with metadata from records dict ignoring given
    keywords.

    **Args:**
      - **records**: dict of records to operate on
    **Kwargs:**
      - **ignore**: Default None, Iterable with keywords of metadata dict to skip.

    **Returns:**
    Pandas DataFrame with Columns as record keys and index as metadatavalue.
    """
    metadata = pd.DataFrame()
    for key, record in records.items():
        rmd = record.metadata.copy()
        if ignore:
            for elm in ignore:
                rmd.pop(elm)
        metadata[key] = pd.Series(rmd)
    return metadata


def _get_base(record_entrie, records):
    """Function to read base entries in configuration dict."""
    base_dict = record_entrie.get('base')
    if base_dict:
        # Allows for easy definition of base by passing {base: name}
        if isinstance(base_dict, str):
            base_dict = {'name': base_dict}

        # Allows singe value baselines
        if isinstance(base_dict, int) or isinstance(base_dict, float):
            return base_dict

        # Pop base name, so the rest can be passed to select
        base_name = base_dict.pop('name')

        base_dict.setdefault('frame_med', True)
        base_dict.setdefault('prop', 'rawData')
        # Needs to be set so all pixels get set by default.
        base_dict.setdefault('roi_pixel', slice(None))

        base = records[base_name]
        base = base.select(
            **base_dict
        )
        return base


def _get_norm(record_entrie, records):
    """Get norm from norm entrie."""
    norm_dict = record_entrie.get('norm')
    if norm_dict:
        # Allows to have the simple config with norm: name
        if isinstance(norm_dict, str):
            norm_dict = {'name': norm_dict}

        # pop name so we use it to select the record
        norm_record = norm_dict.pop('name')

        # Set default kwargs for the select
        norm_dict.setdefault('prop', 'basesubed')
        norm_dict.setdefault('frame_med', True)
        # Using all pixels will make it allways work if same camera is used.
        norm_dict.setdefault('roi_pixel', slice(None))

        norm = records[norm_record]
        norm = norm.select(
            **norm_dict
        )
        return norm


def import_relational(record_entrie, records):
    """Import relational record configuration.

    A relational record configureation is when records are first
    importet via a batch import and then they are assinged das data
    or base or norm of a resulting recrod

    **Arguments**
      - **record_entrie**
        a dict defining the relations between the differenct records
      - **records**
        a dict with named records.

    **Returns**
    dict of records
    """
    name = record_entrie['name']
    rawData = record_entrie['rawData']
    logger.info('Importing {}'.format(name))

    # Allows to have a single string as rawData
    if isinstance(rawData, str):
        record = records[rawData].copy()
    # Allows to have a list of strings as rawData
    else:
        rawDataRecords = [records[elm] for elm in rawData]
        record = core.concatenate_list_of_SfgRecords(rawDataRecords)

    kwargs_record = record_entrie.get('kwargs_record')
    if kwargs_record:
        for key, value in kwargs_record.items():
            logger.debug(
                'Setting {} to {} for record {}'.format(key, value, record)
            )
            try:
                setattr(record, key, value)
            except:
                logger.warn(
                    'Cant set {} to {} for {}'.format(key, value, record)
                )

    base = _get_base(record_entrie, records)
    if not isinstance(base, type(None)):
        record.base = base

    norm = _get_norm(record_entrie, records)
    if not isinstance(norm, type(None)):
        record.norm = norm

    return record


def import_record(record_entrie, records):
    """Import of a single record via given record_entrie dict.
    and lookup already import records within records
    """
    logger.info('Importing {}'.format(record_entrie['name']))
    fpath = record_entrie['fpath']
    kwargs_record = record_entrie.get('kwargs_record', {})

    base = _get_base(record_entrie, records)
    if not isinstance(base, type(None)):
        kwargs_record['base'] = base

    norm = _get_norm(record_entrie, records)
    if not isinstance(norm, type(None)):
        kwargs_record['norm'] = norm

    record = core.SfgRecord(fpath, **kwargs_record)
    return record


def import_records(config_records):
    """Import records

    **Kwargs:**
      - **relations**: If given use relations imports per record.
    """

    records = {}
    for record_entrie in config_records:
        record = import_record(record_entrie, records)

        # Update record name with its real record
        records[record_entrie['name']] = record

    return records


def set_relations(config_records, records):
    """Set relational imports.

    This runs the complete relation import config.
    """
    ret = {}
    for record_entrie in config_records:
        name = record_entrie['name']
        all_records = {**records, **ret}
        ret[name] = import_relational(record_entrie, all_records)
    return ret


def make_models(config_models, records, save_models=True, config_models_path='./models.yaml', clear=True):
    """Make data models, aka. fits.
    **Arguments:**
    - **config_models**: dict with configuration for models
    - **records**: Dict of records that models are piked from

    **kwargs:**
    - **save_models**: Optional, update models file on hdd with result
    - **clear**: clear fitargs fromd default values

    **Returns:**
    list of model objects.
    """
    models = {}

    logger.info('Making Models...')
    for model_name in sort(list(config_models.keys())):
        logger.info('Working on model {}'.format(model_name))
        this_model_config = config_models[model_name]
        # Replace record string with real record becuse real records contain the data
        record_name = this_model_config['record']
        this_model_config['record'] = records[record_name]

        model = sfg2d.models.model_fit_record(**this_model_config)
        models[model_name] = model

        # Clear fitargs from default values. They do no harm but clutter update
        # the models file and make it hard to read.
        this_fitarg = model.fitarg
        if clear:
            this_fitarg = clear_fitarg(this_fitarg)
        # Update kwargs with fit results so the results are available
        dpath.util.set(this_model_config, 'kwargs_model/fitarg', this_fitarg)
        #setback record name to string
        this_model_config['record'] = record_name

    # Update models on disk because we want the fit results to be saved
    old_models = {}
    with open(config_models_path, 'r') as models_file:
        old_models = yaml.load(models_file)

    try:
        new_models = {**old_models, **config_models}
    except TypeError:
        logger.warn('Replacing old models with new models due to error')
        new_models = config_models

    if save_models:
        with open(config_models_path, 'w') as models_file:
            logger.info('Saving models to {}'.format(
                os.path.abspath(config_models_path))
            )
            yaml.dump(new_models, models_file, default_flow_style=False)

    # Update config_models with fit results
    config_models = new_models

    return models


def cache_records(records, cache_dir=CACHE_DIR[1]):
    """Save a cached version of the records in .npz files in cache folder."""
    try:
        os.mkdir(cache_dir)
        logger.info('Create cachedir: {}'.format(cache_dir))
    except FileExistsError:
        pass

    for key, record in records.items():
        fname = cache_dir + '/' + key
        logger.debug('Saving cached record to {}'.format(fname))
        record.save(fname)


def read_cache_list(fnames):
    """
    **Args:**
      - **fnames**: List of filenames
    """
    ret = {}
    for fname in fnames:
        key = os.path.basename(fname).split('.')[0]
        ret[key] = sfg2d.SfgRecord(fname)
    return ret


def read_cache(cache_dir=CACHE_DIR[1]):
    """Read all the .npz files in cache_dir and return dict of them.

    **Kwargs:**
      - **cache_dir**: String with dir to read files from

    **Returns:**
    Dictinary with read spectra. The filenames are used as keys.
    """

    fnames = glob(cache_dir + '/*.npz')
    if len(fnames) <= 0:
        raise ValueError('{} contains no .npz files.'.format(cache_dir))
    ret = {}
    for fname in fnames:
        key = os.path.basename(fname).split('.')[0]
        ret[key] = sfg2d.SfgRecord(fname)
    return ret


def clear_fitarg(fitarg):
    """Clear default values from fitarg dict."""
    ret = fitarg.copy()
    for key, value in fitarg.items():
        if key.startswith('limit_') and value==None:
            ret.pop(key)
        if key.startswith('fix_') and value==False:
            ret.pop(key)
    return ret
