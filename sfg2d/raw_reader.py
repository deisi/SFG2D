"""Module to read raw data and produce records with."""
from pylab import *
import pip
import sys
import os
import numpy as np
import sfg2d.core as core
import sfg2d.fig as fig
import sfg2d.models
import dpath.util
import logging
from . import myyaml as yaml
import pandas as pd
plt.ion()

logging.basicConfig(level=logging.DEBUG)
gitversion = '.gitversion'

def main(config_file='./raw_config.yaml'):
    global records, figures, configuration, models
    records = {}
    figures = {}
    configuration = {}
    models = {}
    config_file = os.path.expanduser(config_file)
    dir = os.path.split(os.path.abspath(config_file))[0]
    cur_dir = os.getcwd()
    os.chdir(dir)
    logging.info('Changing to: {}'.format(dir))

    # Import the configuration
    with open(config_file) as ifile:
        configuration = yaml.load(ifile)

    options = configuration.get('options')
    # Import global options for this data set
    if options:
        read_options(options)

    # Import and configure data
    records = import_records(configuration['records'])

    # Make Models
    config_models = configuration.get('models')
    if config_models:
        logging.info('Making Models...')
        models = make_models(config_models)

    # Make figures
    figures_config = configuration.get('figures')
    if figures_config:
        figures = make_figures(figures_config)

    # Export a pdf with all figures
    list_of_figures = [figures[key][0] for key in sorted(figures.keys())]
    fig.save_figs_to_multipage_pdf(list_of_figures, './figures_all.pdf')

    # Write down the used git version so we can go back
    try:
        from git import Repo, InvalidGitRepositoryError
        module_path = core.__file__
        repo = Repo(module_path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        with open(dir + '/' + gitversion, 'r+') as ofile:
            if sha not in ofile.read():
                logging.info('Appending {} to {}'.format(
                    sha, os.path.abspath(gitversion)))
                ofile.write(sha + '\n')
    except InvalidGitRepositoryError:
        logging.warning('Cant Save gitversion because no repo available.')

    # Write down all installed python modules
    installed_packages = pip.get_installed_distributions()
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
         for i in installed_packages])
    with open(dir + '/.installed_packages', 'w') as ofile:
        for package in installed_packages_list:
            ofile.write(package+'\n')


def read_options(options):
    """Read and apply options."""
    #global configuration
    mplstyle = options.get('mplstyle')
    if mplstyle:
        import matplotlib.pyplot as plt
        plt.style.use(mplstyle)

    file_calib = options.get('file_calib')
    if file_calib:
        calib_pixel = np.loadtxt(file_calib)
        try:
            wavelength = calib_pixel.T[1]
            for config_record in configuration['records']:
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
            for config_record in configuration['records']:
                dpath.util.new(
                    config_record, 'kwargs_record/wavenumber', wavenumber)
        except IndexError:
            logging.warning('Cant find wavenumber in calib file %s'.format(
                    file_calib))

    cache_dir = options.get('cache_dir', './cache')
    for config_record in configuration['records']:
        config_record.setdefault('cache_dir', cache_dir)


def import_records(config_records):
    """Import records"""
    records = {}
    for record_entrie in config_records:
        logging.info('Importing {}'.format(record_entrie['name']))
        fpath = record_entrie['fpath']
        kwargs_record = record_entrie.get('kwargs_record', {})
        base_dict = record_entrie.get('base')
        if base_dict:
            base = records[base_dict['name']].select(
                prop='rawData',
                frame_med=True
            )
            kwargs_record['base'] = base

        norm_dict = record_entrie.get('norm')
        if norm_dict:
            norm = records[norm_dict['name']].select(
                prop='basesubed',
                frame_med=True
            )
            kwargs_record['norm'] = norm

        #kwargs_record.setdefault('wavelength', wavelength)
        #kwargs_record.setdefault('wavenumber', wavenumber)

        record = core.SfgRecord(fpath, **kwargs_record)
        # Update record name with its real record
        records[record_entrie['name']] = record

        # Save cached version of record
        fname = record_entrie['cache_dir'] + '/' + record_entrie['name'] + '.npz'
        logging.info('Saving cached record in {}'.format(os.path.abspath(fname)))
        #record.save(fname)

    return records

def make_models(config_models):
    """Make data models, aka. fits.

    **Returns:**
    list of model objects.
    """
    models = {}
    for model_name in sort(list(config_models.keys())):
        logging.info('Working on model {}'.format(model_name))
        this_model_config = config_models[model_name]
        # Replace record string with real record becuse real records contain the data
        record_name = this_model_config['record']
        this_model_config['record'] = records[record_name]

        model = sfg2d.models.model_fit_record(**this_model_config)
        models[model_name] = model

        # Update kwargs with fit results so the results are available
        dpath.util.set(this_model_config, 'kwargs_model/fitarg', model.fitarg)
        #setback record name to string
        this_model_config['record'] = record_name

    # Update models on disk because we want the fit results to be saved
    fname_models_file = 'models.yaml'
    with open(fname_models_file, 'r') as models_file:
        old_models = yaml.load(models_file)

    try:
       new_models = {**old_models, **config_models}
    except TypeError:
        logging.warn('Replacing old models with new models due to error')
        new_models = config_models

    with open(fname_models_file, 'w') as models_file:
        logging.info('Saving models to {}'.format(os.path.abspath(fname_models_file)))
        yaml.dump(new_models, models_file)

    # Update config_models with fit results
    config_models = new_models

    return models

def make_figures(config_figures):
    """Make the figures.

    **Arguments:**
      - **config_figures**
        list of dictionary with figure configurations.

    **Returns:**
    A list of tuples with [(figures, axes), () , ...]

    """
    figures = {}
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
            dpath.util.set(kwargs_fig, path, records[record_name])

        # Replace model strings with real models
        found_models = dpath.util.search(
            kwargs_fig, '**/model', yielded=True
        )
        for path, model_name in found_models:
            logging.info("Configuring {} with {}".format(path, model_name))
            dpath.util.set(kwargs_fig, path, models[model_name])


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


def get_pd_fitargs():
    """Return fitargs as DataFrame with model names."""
    serach_res = dpath.util.search(
        configuration['models'],
        '*/kwargs_model/fitarg',
        yielded=True
    )
    models = [path.split('/')[0] for path, _ in serach_res]
    datas = [fitarg for _, fitarg in serach_res]

    df = pd.DataFrame.from_dict(data)
    df['model'] = models
    return df
