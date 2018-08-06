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

logging.basicConfig(level=logging.INFO)

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


class Analyser():
    def __init__(self, config_file='./raw_config.yaml'):
        self.config_file = os.path.expanduser(config_file)
        self._configuration={}
        self._records = {}
        self._models = {}
        self._figures = {}

        os.chdir(self.dir)
        logging.info('Changing to: {}'.format(self.dir))

    def __call__(
            self,
            get_configuration=True,
            apply_options=True,
            import_records=True,
            make_models=True,
            save_cache=True,
            make_figures=True,
            save_figures=True,
            save_gitversion=True,
            save_packages=True,
    ):
        """General anlysis call."""

        # Import the configuration
        if get_configuration:
            self._configuration = self.get_configuration()

        # Import global options for this data set
        if apply_options:
            self.apply_options()

        # Import and configure data
        if import_records:
            self._records = self.import_records()

        # Make Models
        if make_models:
            self._models = self.make_models()

        # Save cached version of records
        if save_cache:
            self.cache_records()

        # Make figures
        if make_figures:
            self._figures = self.make_figures()

        # Export a pdf with all figures
        if save_figures:
            self.save_figs()

        # Writing down the used gitversion helps runing code later.
        if save_gitversion:
            self.save_gitversion()

        # Write down all installed python modules
        if save_packages:
            self.save_packages()

    @property
    def configuration(self):
        return self._configuration

    @property
    def dir(self):
        return os.path.split(os.path.abspath(self.config_file))[0]

    @property
    def records(self):
        return self._records

    @property
    def models(self):
        return self._models

    @property
    def figures(self):
        return self._figures

    @property
    def options(self):
        return self.configuration.get(OPTIONS, {})

    @property
    def cache_dir(self):
        return self.options.get(*CACHE_DIR)

    @property
    def figure_file(self):
        return self.options.get(*FIGURE_FILE)

    def get_configuration(self):
        with open(self.config_file) as ifile:
            configuration = yaml.load(ifile)
        return configuration

    def update_configuration(self):
        self._configuration = self.get_configuration()

    def update_models(self, *args):
        self.update_configuration()
        self._models = self.make_models(*args)

    def update_figures(self):
        self.update_configuration()
        self._figures = self.make_figures()

    def apply_options(self):
        """Read and apply options."""
        options = self.options
        if options == {}:
            return

        mplstyle = options.get(MPLSTYLE)
        if mplstyle:
            import matplotlib.pyplot as plt
            logging.info('Applying mplstyle {}'.format(mplstyle))
            plt.style.use(mplstyle)

        file_calib = options.get(*FILE_CALIB)
        if file_calib:
            calib_pixel = np.loadtxt(file_calib)
            try:
                wavelength = calib_pixel.T[1]
                for config_record in self.configuration[RECORDS]:
                    dpath.util.new(
                        config_record, 'kwargs_record/wavelength', wavelength)
            except IndexError:
                logging.warning("Cant find wavelength in calib file %s".format(
                    file_calib))
            try:
                wavenumber = calib_pixel.T[2]
                #dpath.util.set(
                #    configuration,
                #    'records/*/kwargs_record/wavenumber',
                #    wavenumber
                #)
                for config_record in self.configuration[RECORDS]:
                    dpath.util.new(
                        config_record, 'kwargs_record/wavenumber', wavenumber)
            except IndexError:
                logging.warning('Cant find wavenumber in calib file %s'.format(
                        file_calib))

        # Makes it possible to define a default vis_wl for all records
        vis_wl = options.get(*VIS_WL)
        if vis_wl:
            for config_record in self.configuration[RECORDS]:
                dpath.util.new(
                    config_record, 'kwargs_record/vis_wl', vis_wl)


    def import_records(self):
        """Import records"""

        config_records = self.configuration[RECORDS]
        records = {}
        for record_entrie in config_records:
            logging.info('Importing {}'.format(record_entrie['name']))
            fpath = record_entrie['fpath']
            kwargs_record = record_entrie.get('kwargs_record', {})
            base_dict = record_entrie.get('base')
            if base_dict:
                # Allows for easy definition of base by passing {base: name}
                if isinstance(base_dict, str):
                    base_dict = {'name': base_dict}

                # Pop base name, so the rest can be passed to select
                base_name = base_dict.pop('name')

                base_dict.setdefault('frame_med', True)
                base_dict.setdefault('prop', 'rawData')

                base = records[base_name]
                base = base.select(
                    **base_dict
                )
                kwargs_record['base'] = base

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

                norm = records[norm_record]
                norm = norm.select(
                    **norm_dict
                )
                kwargs_record['norm'] = norm

            #kwargs_record.setdefault('wavelength', wavelength)
            #kwargs_record.setdefault('wavenumber', wavenumber)

            record = core.SfgRecord(fpath, **kwargs_record)
            # Update record name with its real record
            records[record_entrie['name']] = record

        return records

    def make_models(self, save_models=True, clear=True):
        """Make data models, aka. fits.

        **kwargs:**
        - **config_models**: Optional, dict with configuration for models
        - **save_models**: Optional, update models file on hdd with result
        - **clear**: Clears fitargs from default values

        **Returns:**
        list of model objects.
        """
        models = {}
        try:
            config_models = self.configuration.get(MODELS).copy()
        except AttributeError:
            logging.info('No modules definded. Skipping')
            return models

        logging.info('Making Models...')
        for model_name in sort(list(config_models.keys())):
            logging.info('Working on model {}'.format(model_name))
            this_model_config = config_models[model_name]
            # Replace record string with real record becuse real records contain the data
            record_name = this_model_config['record']
            this_model_config['record'] = self.records[record_name]

            model = sfg2d.models.model_fit_record(**this_model_config)
            models[model_name] = model

            this_fitarg = model.fitarg
            if clear:
                this_fitarg = clear_fitarg(this_fitarg)
            # Update kwargs with fit results so the results are available
            dpath.util.set(this_model_config, 'kwargs_model/fitarg', this_fitarg)
            #setback record name to string
            this_model_config['record'] = record_name

        # Update models on disk because we want the fit results to be saved
        old_models = {}
        with open(MODEL_FILE, 'r') as models_file:
            old_models = yaml.load(models_file)

        try:
            new_models = {**old_models, **config_models}
        except TypeError:
            logging.warn('Replacing old models with new models due to error')
            new_models = config_models

        if save_models:
            with open(MODEL_FILE, 'w') as models_file:
                logging.info('Saving models to {}'.format(os.path.abspath(MODEL_FILE)))
                yaml.dump(new_models, models_file, default_flow_style=False)

        # Update config_models with fit results
        config_models = new_models

        return models

    def cache_records(self):
        """Save a cached version of the records in .npz files in cache folder."""
        try:
            os.mkdir(self.cache_dir)
            logging.info('Create cachedir: {}'.format(self.cache_dir))
        except FileExistsError:
            pass

        for key, record in self.records.items():
            fname = self.cache_dir + '/' + key
            logging.debug('Saving cached record to {}'.format(fname))
            record.save(fname)


    def make_figures(self):
        """Make the figures.

        **Arguments:**
        - **config_figures**
            list of dictionary with figure configurations.

        **Returns:**
        A list of tuples with [(figures, axes), () , ...]

        """

        # Deepcopy to allow for independent nested dicts
        config_figures = deepcopy(self.configuration.get(FIGURES))
        figures = {}
        if not config_figures:
            return figures

        for fig_config in config_figures:
            # Name is equal the configuration key, so it must be stripped
            fig_name, fig_config = list(fig_config.items())[0]
            logging.info('Making: {}'.format(fig_name))
            fig_type = fig_config['type']
            kwargs_fig = fig_config['kwargs'].copy()

            # Replace records strings with real records:
            found_records = dpath.util.search(
                kwargs_fig, '**/record', yielded=True
            )
            for path, record_name in found_records:
                logging.info("Configuring {} with {}".format(path, record_name))
                dpath.util.set(kwargs_fig, path, self.records[record_name])

            # Replace model strings with real models
            found_models = dpath.util.search(
                kwargs_fig, '**/model', yielded=True
            )
            for path, model_name in found_models:
                logging.info("Configuring {} with {}".format(path, model_name))
                dpath.util.set(kwargs_fig, path, self.models[model_name])


            fig_func = getattr(fig, fig_type)
            # Use fig_name as default figure num
            try:
                dpath.util.get(kwargs_fig, 'kwargs_figure/num')
            except KeyError:
                if not kwargs_fig.get('kwargs_figure'):
                    dpath.util.new(kwargs_fig, 'kwargs_figure/num', fig_name)
                else:
                    kwargs_fig['kwargs_figure']['num'] = fig_name

            figures[fig_name] = fig_func(**kwargs_fig)
        return figures

    def save_figs(self):
        list_of_figures = [self.figures[key][0] for key in sorted(self.figures.keys())]
        fig.save_figs_to_multipage_pdf(list_of_figures, self.figure_file)

    def save_gitversion(self):
        try:
            from git import Repo, InvalidGitRepositoryError
            module_path = core.__file__
            repo = Repo(module_path, search_parent_directories=True)
            sha = repo.head.object.hexsha
            gitfile = self.dir + '/' + GITVERSION
            # Need to make shure the file exists, in case it was not created.
            open(gitfile, 'a').close()
            with open(gitfile, 'r+') as ofile:
                if sha not in ofile.read():
                    logging.info('Appending {} to {}'.format(
                        sha, os.path.abspath(GITVERSION)))
                    ofile.write(sha + '\n')
        except InvalidGitRepositoryError:
            logging.warning('Cant Save gitversion because no repo available.')

    def save_packages(self):
        installed_packages = get_installed_distributions()
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
            for i in installed_packages])
        with open(self.dir + '/' + INSTALLED_PACKAGES, 'w') as ofile:
            for package in installed_packages_list:
                ofile.write(package+'\n')


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

def import_records(config_records):
    """Import records"""

    records = {}
    for record_entrie in config_records:
        logging.info('Importing {}'.format(record_entrie['name']))
        fpath = record_entrie['fpath']
        kwargs_record = record_entrie.get('kwargs_record', {})
        base_dict = record_entrie.get('base')
        if base_dict:
            # Allows for easy definition of base by passing {base: name}
            if isinstance(base_dict, str):
                base_dict = {'name': base_dict}

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
            kwargs_record['base'] = base

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
            kwargs_record['norm'] = norm

        #kwargs_record.setdefault('wavelength', wavelength)
        #kwargs_record.setdefault('wavenumber', wavenumber)

        record = core.SfgRecord(fpath, **kwargs_record)
        # Update record name with its real record
        records[record_entrie['name']] = record

    return records


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

    logging.info('Making Models...')
    for model_name in sort(list(config_models.keys())):
        logging.info('Working on model {}'.format(model_name))
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
        logging.warn('Replacing old models with new models due to error')
        new_models = config_models

    if save_models:
        with open(config_models_path, 'w') as models_file:
            logging.info('Saving models to {}'.format(
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
        logging.info('Create cachedir: {}'.format(cache_dir))
    except FileExistsError:
        pass

    for key, record in records.items():
        fname = cache_dir + '/' + key
        logging.debug('Saving cached record to {}'.format(fname))
        record.save(fname)


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
