import os
import re
import sys
import yaml
import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np

import sfg2d

RECORDS = 'records'
OPTIONS = 'options'
PRESET = 'preset'
PLOTS = 'plots'
FIT = 'fit'
SPEC = 'spec'
TRACE = 'trace'
NAME = 'name'
BASE = 'base'
NORM = 'norm'
TYPE = 'type'
SUBSELECT = 'subselect'
PLOT_TYPES = [SPEC, TRACE]

with open("analyse.yaml", 'r') as stream:
    try:
        global_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit()

options = global_config.get('options', {})
figure_path = options.get('figure_path', './figures')
cache_path = options.get('cache_path', './cache')
ext = options.get('savefig', {}).get('ext', '.pdf')

# Presets for plots.
PRESETS_DICT = {
    'spec':
    {
        'subselect':
        {
            'x_property': 'wavenumber',
            'y_property': 'normalized',
            'frame_med': True,
        },
        'plot':
        {
            'xlabel': 'Wavenumber in 1/cm'
        }
    },
    'baseline':
    {
        'subselect':
        {
            'x_property': 'pixel',
            'frame_med': True,
            'y_property': 'rawData',
        },
        'plot':
        {
            'title': 'Baseline',
            'xlabel': 'Pixel',
            'ylabel': 'Counts',
        }
    },
    'bleach':
    {
        'subselect': {
            'x_property': 'wavenumber',
            'y_property': 'bleach_abs',
            'frame_med': True,
        },
        'plot': {
            'title': 'Bleach',
            'xlabel': 'Wavenumber in 1/cm',
            'ylabel': 'Counts',
        }
    },
    'crosscorrelation':
    {
        'subselect': {
            'x_property': 'wavenumber',
            'y_property': 'basesubed',
            'frame_med': True,
        },
        'plot': {
            'title': 'Crosscorrelation',
            'xlabel': 'Wavenumber in 1/cm',
            'ylabel': 'Counts',
        }
    },
    'trace':
    {
        'subselect':
        {
            'x_property': 'pp_delays',
            'y_property': 'basesubed',
            'frame_med': True,
            'pixel_mean': True,
        },
        'plot':
        {
            'title': 'Trace',
            'xlabel': 'Time in fs',
            'marker': 'o',
        }
    }
}

def parse_config(sub_config, top_config, keys=('subselect', 'fig', 'plot', 'ax')):
    """This parser fills the defaults from top_config,
    into sub_config and doesnt overwrite any of them."""

    ret = sub_config.copy()
    for key in keys:
        value = ret.get(key)
        if value:
            value.update(top_config.get(key, {}))
        else:
            ret[key] = top_config.get(key, {})
    return ret

def configure_plot(config, name, defaults={}):
    """Use config dict and configure the named element with given defaults."""
    ret = defaults.copy()
    subselect_kw = collect_config(config, name, 'subselect')
    subselect_kw = translate_slice(subselect_kw)
    plot_kw = collect_config(config, name, 'plot')
    ax_kw = collect_config(config, name, 'ax')
    fig_kw = collect_config(config, name, 'fig')

    ret.get('subselect', {}).update(subselect_kw)
    ret.get('plot', {}).update(plot_kw)
    ret.get('ax', {}).update(ax_kw)
    ret.get('fig', {}).update(fig_kw)
    return ret


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def _map_slice(slice_string):
    """map a string of 'slice(start, stop, step)' into a python slice.

    if wrong string is gi"""
    match = re.search('slice\((.+)\)', slice_string)
    if not match:
        raise InputError(
            'Slice_string {} must be string of the form slice(start, stop, step)'.format(slice_string)
        )
    return slice(*[int(number) for number in match.group(1).split(",")])


def set_default_value(dict, key, value):
    """Set key to value in dict if key does not exist."""
    if not dict.get(key):
        dict[key] = value


def translate_slice(config):
    """Read the subselect kwgs.

    Returns:
    A translated version of the config, with slices
    replaced by slices.

    """
    # Replace rois slice strings with python slices.
    for name, value in config.items():
        if "roi_" in name:
            if 'slice' in value:
                config[name] = _map_slice(value)
        if "rois_" in name:
            rois = []
            for sub_roi in value:
                if 'slice' in sub_roi:
                    rois.append(_map_slice(sub_roi))
            config[name] = rois
    return config


def make_spec_config(config):
    """Make a config for a spectrum taking inheritance into account."""

    # The copy ensures we dont overwrite the config values.
    spec_config = config.get('spec', {}).copy()
    spec_config['plot'] = config.get('plot', {}).copy()
    spec_config['ax'] = config.get('ax', {}).copy()
    spec_config['fig'] = config.get('fig', {}).copy()
    spec_config['subselect'] = translate_slice(config.get('subselect', {}).copy())
    return spec_config


def _fit(fit_config, data, sigma=None):
    """fit_config is a config dict for the fit.
    data is 4d numpy array with [delays, frames, spec, pixel]"""
    def make_fit(xdata, ydata, sigma=None):
        gm = fit_model(xdata, ydata, sigma=sigma, **fit_config.get('attributes', {}))
        gm.curve_fit()
        if fit_config.get('show'):
            kwgs = {'show_data': False, 'show_fit_line': True, 'show_box': True}
            kwgs.update(fit_config.get("plot", {}))
            gm.plot(**kwgs)
            #kwgs = fit_config.get('plot', {})
            #x_sample = np.linspace(xdata.min(), xdata.max(), 300)
            #plt.plot(x_sample, gm.fit_res(x_sample), **kwgs)
        return gm

    xdata, ydata = data
    fit_model = getattr(sfg2d.models, fit_config['model'])
    attr = fit_config.get('attributes')
    # Find aligning axis between xdata and ydata
    if xdata.shape[0] not in ydata.shape:
        raise InputError(None, 'Can not fit this data shape.')

    mask = [xdata.shape[0] == elm for elm in ydata.shape]
    if np.count_nonzero(mask) > 1:
        raise InputError(None, 'No  unique connection between xdata and ydata')

    matching_axis = np.where(mask)
    ret = []
    # Spectra data
    if matching_axis[0][0] == 3:
        for delay in ydata:
            for frame in delay:
                for spec in frame:
                    ret.append(make_fit(xdata, spec))
    # Trace data
    elif matching_axis[0][0] == 0:
        y = ydata.T
        for pixel in y:
            for spec in pixel:
                for frame in spec:
                    if isinstance(sigma, type(None)):
                        ret.append(make_fit(xdata, frame))
                    elif frame.shape == sigma.shape:
                        print("Shapes for errors match")
                        ret.append(make_fit(xdata, frame, sigma))
                    else:
                        print("Shapes for dont match")
                        ret.append(make_fit(xdata, frame))

    else:
        raise NotImplementedError
    return ret

def subselect(config, defaults={}):
    """Subselect from records with with given config and defaults.
    config updates defaults.

    returns a sfg2d.Record obj.
    """
    defaults.update(config)
    record = records[defaults[NAME]]
    # Remember that this takes the roi and rois attributes if an attribute
    # is not given.
    return record.subselect(**defaults[SUBSELECT])


def import_record(config, attributes):
    """Process all records entires.

    config: The config dict of ths record,
    attributes: Global attributes.
    """
    if config.get('skip'):
        print('Skipping Record Import')
        return

    fpath = config['fpath']
    if isinstance(fpath, str):
        fpath = [fpath]
    record = sfg2d.core.SfgRecords_from_file_list(fpath)

    config_of_base = config.get(BASE)
    if config_of_base:
        # The specific config must update the default config.
        default_settings = {'subselect': {'frame_med': True}}
        record.base = subselect(config_of_base, default_settings)[1]

    config_of_norm = config.get(NORM)
    if config_of_norm:
        # We want the specific config to update the default config.
        default_settings = {'subselect':
                            {'frame_med': True, 'y_property': 'basesubed'}}
        record.norm = subselect(config_of_norm, default_settings)[1]

    # Global Attributes:
    for name, value in attributes.items():
        setattr(record, name, value)

    # Replace broken pixel
    if global_config.get('replace'):
        for elm in global_config['replace']:
            print('Replacing Pixel: ', elm['pixel'])
            sfg2d.replace_pixel(record, elm['pixel'], elm.get('region', 5))

    # Attributes per record
    config_of_attr = config.get('attributes')
    if config_of_attr:
        config_of_attr = translate_slice(config_of_attr)
        for name, value in config_of_attr.items():
            setattr(record, name, value)

    cache = config.get('cache', True)
    if cache:
        fpath = '{}/{}_pump{}'.format(
            cache_path, config['name'], record.pump_freq
        )
        print('Saving Record to:')
        print(fpath)
        record.save(fpath)
    return record

def _plot_record(plot_config, data):
    """
    Make a plot of data.

    config: Nested dict with: subselection, plot, ax and fig as keys.
    data: 4d numpy array with [pump, frame, spec, pixel]

    """
    fig_kw = plot_config.get('fig', {})
    ax_kw = plot_config.get('ax', {})
    plot_kw = plot_config.get('plot', {})
    if not fig_kw.get('num'):
        fig_kw['num'] = plot_config.get('name')
    fig = plt.figure(**fig_kw)
    ax = plt.axes(**ax_kw)
    subselect_kw = plot_config.get('subselect', {})
    if subselect_kw['x_property'] == 'pixel':
        set_default_value(plot_kw, 'xlabel', 'Pixel')
    if subselect_kw['x_property'] == 'wavelength':
        set_default_value(plot_kw, 'xlabel', 'Wavelength')
    if subselect_kw['x_property'] == 'wavenumber':
        set_default_value(plot_kw, 'xlabel', 'Wavenumber')
    if subselect_kw['x_property'] == 'pp_delays':
        set_default_value(plot_kw, 'xlabel', 'Pump-Probe Delay in fs')
    if subselect_kw['x_property'] == 'frames':
        set_default_value(plot_kw, 'xlabel', 'Frame Number')
    if plot_kw.get('title'):
        ax.set_title(plot_kw.pop('title'))
    if plot_kw.get('xlabel'):
        ax.set_xlabel(plot_kw.pop('xlabel'))
    if plot_kw.get('ylabel'):
        ax.set_ylabel(plot_kw.pop('ylabel'))
    xdata, ydata = data
    if subselect_kw['x_property'] in ('pixel', 'wavelength', 'wavenumber'):
        sfg2d.plotting.plot_spec(xdata, ydata, ax, **plot_kw)
    elif subselect_kw['x_property'] == 'pp_delays':
        sfg2d.plotting.plot_trace(xdata, ydata, ax, **plot_kw)
    else:
        raise NotImplementedError
    return fig

def _subselect_plot(record, config):
    """The default plot that uses subselection on data."""

    preset = config.get(PRESET, config[TYPE])
    plot_config = PRESETS_DICT[preset].copy()
    for key in ('subselect', 'fig', 'ax', 'plot', 'name'):
        value = config.get(key)
        if value:
            try:
                plot_config[key].update(value)
            except KeyError:
                plot_config[key] = value
    #print('Plot Config:\n{}'.format(plot_config))
    plot_config[SUBSELECT] = translate_slice(plot_config[SUBSELECT])
    data = record.subselect(**plot_config[SUBSELECT])
    # Plot the spectrum
    fig = _plot_record(plot_config, data)

    # fit the data.
    sigma = None
    fit = config.get(FIT)
    models = []
    fit_data = data
    if fit:
        # Get errors for errorbars spagetti code
        if fit.get('error'):
            subs = plot_config.get('subselect', {}).copy()
            # Because we need the frames to calculate an standard error.
            subs['frame_med'] = False
            y = record.subselect(
                **subs,
            )[1]
            sigma = sem(y, 1).squeeze()

        ## Hack to subslice fit data
        if fit.get('roi_delay'):
            print('Warning: Hacky and shaky delay roiing for fit:')
            print(sigma)
            roi_delay = _map_slice(fit.pop('roi_delay'))
            fit_data = data[0][roi_delay], data[1][roi_delay]
            sigma = sigma[roi_delay]
        fit_config = parse_config(fit, plot_config, keys=('subselect', 'fig'))
        print('Fit Config:\n{}'.format(fit_config))
        models = _fit(fit_config, fit_data, sigma)

    cache = config.get('cache')
    if cache:
        save_dict = {}
        save_dict['xdata'] = data[0].tolist()
        save_dict['ydata'] = data[1].squeeze().tolist()
        if fit:
            save_dict['xfit'] = fit_data[0].tolist()
            save_dict['yfit'] = fit_data[1].squeeze().tolist()
            if not isinstance(sigma, type(None)):
                save_dict['sigma'] = sigma.tolist()
                save_dict['fit_cov'] = models[0].cov.tolist()
            save_dict['p'] = models[0].p.tolist()
        savetxt = cache.get('savetxt')
        if savetxt:
            default_name = '{}/{}_{}_{}.yaml'.format(
                cache_path, record_config[NAME], config[TYPE], config[NAME]
            )
            print('Saving Cache to:')
            print(default_name)
            with open(default_name, "w") as fname:
                yaml.dump(save_dict, fname)

    if not config.get('show'):
        plt.close(fig)
    return fig, models


def _record_trace_plots(record, config):
    """Use record.plot_trace to make the plot."""

    fig, ax = plt.subplots()
    plot_trace_kw = config.get('plot_trace', {})
    record.plot_trace(**plot_trace_kw)
    ax.set_xlim(*config.get('xlim', {}))
    ax.set_ylim(*config.get('ylim', {}))
    fit = config.get(FIT)
    if not config.get('show'):
        plt.close(fig)
    return fig, None


def _contour_plot(record, config):
    attr = config.get('attributes')
    if attr:
        attr = translate_slice(attr)
        for name, value in attr.items():
            setattr(record, name, value)

    contour_kw = config.get('contour_kw', {})
    contour_kw.setdefault("show_axb", False)
    contour_kw.setdefault("show_axr", False)
    contour_kw.setdefault("show_axl", False)
    contour_kw.setdefault("N", 50)
    contour_kw.setdefault('pixel_med', 11)
    if contour_kw.get("levels"):
        contour_kw['levels'] = np.linspace(*contour_kw.get("levels"))
    fig, ax, axl, axb, axr = sfg2d.plotting.contour(record, **contour_kw)
    if config.get('xlim'):
        ax.set_xlim(config.get('xlim'))
    if config.get('ylim'):
        ax.set_ylim(config.get('ylim'))
    return fig, None


def _frame_track_plot(record, config):
    fig, ax = plt.subplots()
    attr = config.get('attributes')
    if attr:
        attr = translate_slice(attr)
        for name, value in attr.items():
            setattr(record, name, value)
    track_kw = config.get('frame_track_kw', {})
    track_kw.setdefault('y_property', 'basesubed')
    sfg2d.plotting.frame_track(record, ax, **track_kw)
    return fig, None


def _bleach_pdf(record, config):
    attr = config.get('attributes')
    if attr:
        attr = translate_slice(attr)
        for name, value in attr.items():
            setattr(record, name, value)
    sfile = config.get('name')
    bleach_plot_kw = config.get('bleach_plot_kw', {})
    sfg2d.plotting.bleach_plot_pdf(
        record,
        sfile,
        **bleach_plot_kw
    )
    # Hack around
    return plt.subplots()[0], None



def make_plot(record, config):
    # Set attributes per plot.
    # This we use quick and dirty error calculation
    # It can also be used as a default for subselection
    attr = config.get('attributes')
    if attr:
        attr = translate_slice(attr)
        for name, value in attr.items():
            setattr(record, name, value)

    if config[TYPE] == 'record_trace':
        fig, models = _record_trace_plots(record, config)
    elif config[TYPE] == 'contour':
        fig, models = _contour_plot(record, config)
    elif config[TYPE] == 'track':
        fig, models = _frame_track_plot(record, config)
    elif config[TYPE] == 'bleach_pdf':
        fig, models = _bleach_pdf(record, config)
    else:
        fig, models = _subselect_plot(record, config)


    return fig, models


records = {}
figs = {}
fits = {}

# Create some folders
if not os.path.isdir(figure_path):
    os.mkdir(figure_path)
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

for record_config in global_config[RECORDS]:
    record = import_record(record_config, global_config.get('attributes', {}))
    if isinstance(record, type(None)):
        continue
    records[record_config[NAME]] = record
    print('\nRunning: {}'.format(record_config[NAME]))
    figures = []
    models = []
    index = 0
    for plot_config in record_config[PLOTS]:
        if plot_config.get('skip'):
            print('Skipping plot {}'.format(plot_config.get(NAME, index)))
            continue
        print('========================')
        print('Record Config:')
        print(record_config)
        print('Plot Config:')
        print(plot_config)
        print('==============================')
        # parse_config(plot_config, record_config),
        figure, model = make_plot(
            record,
            plot_config
        )
        figures.append(figure)
        models.append(model)
        save_fig_kw = plot_config.get('save_fig')
        default_name = '{}_{}_{}'.format(
            record_config[NAME], plot_config[TYPE], index
        )
        if isinstance(save_fig_kw, bool):
            if save_fig_kw:
                fname = figure_path + '/' + plot_config.get(NAME, default_name) + ext
                print("Saving figure to: ", fname)
                plt.savefig(fname)
        else:
            if save_fig_kw:
                if not save_fig_kw.get('fname'):
                    save_fig_kw['fname'] = save_fig_kw.get('fname',
                                                           default_name)
                print('Saving figure with:')
                print(save_fig_kw)
                fname = save_fig_kw.pop('fname')
                plt.savefig(fname, **save_fig_kw)
        index += 1
    figs[record_config[NAME]] = figures
    fits[record_config[NAME]] = models

plt.show(block=False)
