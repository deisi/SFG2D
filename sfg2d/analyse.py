
# global_config utf-8
import os
import re
import sys
import yaml
import matplotlib.pyplot as plt
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

def parse_config(sub_config, top_config):
    """This parser fills the defaults from top_config,
    into sub_config and doesnt overwrite any of them."""

    ret = sub_config.copy()
    for key in ('subselect', 'fig', 'plot', 'ax'):
        value = ret.get(key)
        if value:
            value.update(top_config.get(key, {}))
        else:
            value = top_config.get(key, {})
        value = sub_config.get(key)
        if value:
            ret[key] = value
    return ret

def configure_plot(config, name, defaults={}):
    """Use config dict and configure the named element with given defaults."""
    ret = defaults.copy()
    subselect_kw = collect_config(config, name, 'subselect')
    subselect_kw = translate_subselect(subselect_kw)
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


def translate_subselect(config):
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
            raise NotImplementedError
    return config


def make_spec_config(config):
    """Make a config for a spectrum taking inheritance into account."""

    # The copy ensures we dont overwrite the config values.
    spec_config = config.get('spec', {}).copy()
    spec_config['plot'] = config.get('plot', {}).copy()
    spec_config['ax'] = config.get('ax', {}).copy()
    spec_config['fig'] = config.get('fig', {}).copy()
    spec_config['subselect'] = translate_subselect(config.get('subselect', {}).copy())
    return spec_config


def _fit(fit_config, data):
    """fit_config is a config dict for the fit.
    data is 4d numpy array with [delays, frames, spec, pixel]"""
    def make_fit(xdata, ydata):
        gm = fit_model(xdata, ydata, fit_config.get('p0'))
        gm.curve_fit()
        if fit_config.get('show'):
            kwgs = {'show_data': False, 'show_fit_line': True, 'show_box': True}
            kwgs.update(fit_config.get("plot", {}))
            gm.plot(**kwgs)
        return gm

    xdata, ydata = data
    fit_model = getattr(sfg2d.models, fit_config['model'])
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
                    print(xdata, frame)
                    ret.append(make_fit(xdata, frame))
    else:
        raise NotImplementedError
    return ret

def _plot_record(plot_config, data):
    """
    Make a plot of data.

    config: Nested dict with: subselection, plot, ax and fig as keys.
    data: 4d numpy array with [pump, frame, spec, pixel]

    """
    fig_kw = plot_config.get('fig', {})
    ax_kw = plot_config.get('ax', {})
    plot_kw = plot_config.get('plot', {})
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
        pass
        sfg2d.plotting.plot_spec(xdata, ydata, ax, **plot_kw)
    elif subselect_kw['x_property'] == 'pp_delays':
        pass
        sfg2d.plotting.plot_trace(xdata, ydata, ax, **plot_kw)
    else:
        raise NotImplementedError
    return fig


def subselect(config, defaults={}):
    """Subselect from records with with given config and defaults.
    config updates defaults.

    returns a sfg2d.Record obj.
    """
    defaults.update(config)
    record = records[defaults[NAME]]
    return record.subselect(**defaults[SUBSELECT])


def import_record(config):
    """Process all records entires."""
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
    return record


def make_plots(record, config):
    type = config[TYPE]
    preset = config.get(PRESET, type)
    plot_config = PRESETS_DICT[preset].copy()
    for key in ('subselect', 'fig', 'ax', 'plot'):
        value = config.get(key)
        if value:
            plot_config[key].update(value)
    print('Plot Config:\n{}'.format(plot_config))
    plot_config[SUBSELECT] = translate_subselect(plot_config[SUBSELECT])
    data = record.subselect(**plot_config[SUBSELECT])
    fig = _plot_record(plot_config, data)

    # fit the data.
    fit = config.get(FIT)
    if fit:
        fit_config = parse_config(fit, plot_config)
        print('Fit Config:\n{}'.format(fit_config))
        _fit(fit_config, data)

    if not config.get('show'):
        plt.close(fig)
    return fig


with open("analyse.yaml", 'r') as stream:
    try:
        global_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit()

records = {}
figs = {}
for record_config in global_config[RECORDS]:
    record = import_record(record_config)
    records[record_config[NAME]] = record
    print('\nRunning: {}'.format(record_config[NAME]))
    for plot_config in record_config[PLOTS]:
        print('========================')
        print(plot_config)
        print(record_config)
        print(parse_config(plot_config, record_config))
        print('==============================')
        figs[record_config[NAME]] = make_plots(
            record,
            parse_config(plot_config, record_config),
        )

plt.show(block=False)
