"""Module to read raw data and produce records with."""
from pylab import *
import yaml
import numpy as np
import sfg2d.core as core
import sfg2d.fig as fig
import dpath.util

fname = 'raw_config.yaml'

plt.ion()


def main():
    global records, figures, config
    records = {}
    figures = {}
    config = {}

    with open(fname) as ifile:
        config = yaml.load(ifile)

    options = config.get('options')
    wavelength = None
    wavenumber = None
    if options:
        mplstyle = options.get('mplstyle')
        if mplstyle:
            import matplotlib.pyplot as plt
            plt.style.use(mplstyle)

        calib_pixel_file = options.get('calib_pixel_file')
        if calib_pixel_file:
            calib_pixel = np.loadtxt(calib_pixel_file)
            try:
                wavelength = calib_pixel.T[1]
            except IndexError:
                print("Cant find wavelength in calib file %s".format(
                    calib_pixel_file))
            try:
                wavenumber = calib_pixel.T[2]
            except IndexError:
                print('Cant find wavenumber in calib file %s'.format(
                    calib_pixel_file))

    for record_entrie in config['records']:
        fpath = record_entrie['fpath']
        record_kwgs = record_entrie.get('record_kwgs', {})
        base_dict = record_entrie.get('base')
        if base_dict:
            base = records[base_dict['name']].select(
                prop='rawData',
                frame_med=True
            )
            record_kwgs['base'] = base

        norm_dict = record_entrie.get('norm')
        if norm_dict:
            norm = records[norm_dict['name']].select(
                prop='basesubed',
                frame_med=True
            )
            record_kwgs['norm'] = norm

        record_kwgs.setdefault('wavelength', wavelength)
        record_kwgs.setdefault('wavenumber', wavenumber)

        if type(fpath) == str:
            record = core.SfgRecord(fpath, **record_kwgs)
        else:
            record = core.SfgRecords_from_file_list(fpath, **record_kwgs)
        records[record_entrie['name']] = record

    for fig_config in config['figures']:
        # Name is equal the config key, so it must be stripped
        fig_name, fig_config = list(fig_config.items())[0]
        print('Making: {}'.format(fig_name))
        fig_type = fig_config['type']
        fig_kwgs = fig_config['fig_kwgs'].copy()

        # Replace records strings with real records:
        found_records = dpath.util.search(fig_kwgs, '**/record',
                                           yielded=True)
        for path, record_name in found_records:
            print("Configuring {} with {}".format(path, record_name))
            dpath.util.set(fig_kwgs, path, records[record_name])

        fig_func = getattr(fig, fig_type)
        figures[fig_name] = fig_func(**fig_kwgs)

    # Finalize
    list_of_figures = [value[0] for value in figures.values()]
    fig.save_figs_to_multipage_pdf(list_of_figures, './figures_all.pdf' )

    try:
        from git import Repo, InvalidGitRepositoryError
        module_path = core.__file__
        repo = Repo(module_path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        with open('.gitversion', 'w') as ofile:
            ofile.write(sha)
    except InvalidGitRepositoryError:
        print('Cant Save gitversion because no repo available.')

