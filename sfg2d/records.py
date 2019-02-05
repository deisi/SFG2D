"""Module for records related function.

A records is a dict of multiple sfg2d.SfgRecord objects that are treated
together.

"""
import os
from . import plot
from . import fig as sfgfig
from .core import SfgRecord
import matplotlib.pyplot as plt
import logging
from numpy import linspace, array, concatenate
from scipy.stats import sem
import pandas as pd

from sfg2d.utils.config import CONFIG
FRAME_AXIS_INDEX = CONFIG['FRAME_AXIS_INDEX']

## Dict of records.
#records = {}
## Dicto of model.
#models = {}

logger = logging.getLogger(__name__)

try:
    if not isinstance(records, dict):
        records = {}
except NameError:
    records = {}

try:
    if not isinstance(models, dict):
        models = {}
except NameError:
    models = {}


def records_agg(names, func):
    """Aggregation function for a selection of records."""
    ret = []
    for name in names:
        record = records[name]
        ret.append(func(record))
    return ret


def records_list_agg(records, func):
    """Aggregation function for a list of records."""
    ret = []
    for record in records:
        ret.append(func(record))
    return ret


def plot_spectra(record_names, kwargs_xdata, kwargs_ydata, kwargs_plots=None, kwargs_xdata_record=None, kwargs_ydata_record=None, kwargs_plot=None):
    """High level function to generate plots.

    **Arguments**
      - **record_names**: List of keys for the `records` dict.
      - **kwargs_xdata**: dict with x kwargs for all records
      - **kwargs_ydata**: dict with y kwargs for all records
      - **kwargs_plots**: dict with kwargs for each record_name
      - **kwargs_xdata_record**: Dict with xdata kwargs per record
        specific kwargs overwrite general
      - **kwargs_ydata_record**: Dict with ydata kwargs per record
        specific kwargs overwrite general
      - **kwargs_plot**: Plot kwargs applied to all lines

    **Returns**:
    The xdatas and ydatas used during making the plot
    """
    global records
    if not kwargs_plots:
        kwargs_plots = {}
    if not kwargs_xdata_record:
        kwargs_xdata_record = {}
    if not kwargs_ydata_record:
        kwargs_ydata_record = {}
    if not kwargs_plot:
        kwargs_plot = {}

    xdatas = []
    ydatas = []
    for name in record_names:
        record = records.get(name)
        if not record:
            print('{} not found in records'.format(name))
            continue

        _kwargs_xdata = kwargs_ydata_record.get(name, {})
        _kwargs_ydata = kwargs_ydata_record.get(name, {})

        tkwx = {**kwargs_xdata, **_kwargs_xdata}
        tkwy = {**kwargs_ydata, **_kwargs_ydata}

        # Must combine global and specific kwarg dicts
        logger.debug('kwargs_plot is: {}'.format(kwargs_plot))
        _kwargs_plot = {**kwargs_plot, **kwargs_plots.get(name, {})}
        xdata = record.select(**tkwx)
        xdatas.append(xdata)
        ydata = record.select(**tkwy)
        ydatas.append(ydata)
        plot.spectrum(xdata, ydata, **_kwargs_plot)
    # Casting to numpy array not allways works, because
    # ydatas can have different shapes
    try:
        xdatas = array(xdatas)
    except ValueError:
        logger.warn('Can cast xdatas of plot_spectra to numpy array.')
    try:
        ydatas = array(ydatas)
    except ValueError:
        logger.warn('Cant cast ydatas of plot_spectra to numpy array.')
    return xdatas, ydatas

def plot_traces(record_names, kwargs_xdata, kwargs_ydata, kwargs_yerr=None, kwargs_plots=None):
    if not kwargs_yerr:
        kwargs_yerr = kwargs_ydata.copy()
        try:
            kwargs_yerr.pop('frame_med')
        except KeyError:
            pass

    if not kwargs_plots:
        kwargs_plots = {}

    for name in record_names:
        record = records.get(name)
        if not record:
            print('{} not found in records'.format(name))
            continue

        kwargs_plot = kwargs_plots.get(name, {})
        xdata = record.select(**kwargs_xdata)
        ydata = record.select(**kwargs_ydata)
        yerr = record.sem(**kwargs_yerr)
        plot.trace(xdata, ydata, yerr=yerr, **kwargs_plot)

def plot_tracks(record_names, kwargs_ydata, kwargs_plots=None):
    if not kwargs_plots:
        kwargs_plots = {}
    for name in record_names:
        record = records.get(name)
        if not record:
            print('{} not found in records'.format(name))
            continue

        kwargs_plot = kwargs_plots.get(name, {})
        ydata = record.select(**kwargs_ydata)
        plot.track(ydata=ydata, **kwargs_plot)

def plot_models(model_names, plot_kwargs=None, text_kwargs=None, kwargs_data=None):

    if not plot_kwargs:
        plot_kwargs = {}

    if not text_kwargs:
        text_kwargs = {}

    for model_name in model_names:
        m = models.get(model_name)
        if not m:
            print('Cant find {} in models'.format(model_name))
            continue

        this_text_kwargs = text_kwargs.get(model_name, {})
        plot_kwarg = plot_kwargs.get(model_name, {})

        if isinstance(kwargs_data, dict):
            kwargs_data.setdefault('fmt', 'o')
            plt.errorbar(m.xdata, m.ydata, m.yerr, **kwargs_data)
        plot_kwarg.setdefault('color', 'r')
        plt.plot(m.xsample, m.ysample, **plot_kwargs)
        if text_kwargs:
            plt.text(s=m.box_str, **this_text_kwargs)


@sfgfig.ioff
def multifig_bleach(record_name, kwargs_xdata=None, kwargs_ydata=None,
             fig_axis=0, kwargs_plot=None,
             sfile='bleach.pdf', ylim=None, titles=None):
    """Function to make a multi figure plot from y data selection.

    `fig_axis` defines the axis of kwargs_ydata selecteion that will be looped
    over. This means by passing fig_axis = 0 you put every pump probe delay
    into a single figure. Of fig_axis = 1 will put each frame into a different
    figure. The result will be exported as pdf into `sfile`. By default all figures
    have the same ylim, that is the maximum amongst all figures.

    **Arguments:**
      - **record_name**: Name of the recrod
    **Kewords:**
      - **kwargs_xdata**: Keywords to select xdata with
      - **kwargs_ydata**: Keywirds to select ydata with
      - **fig_axis**: Axis to loop the figures over. Each entry in ydata of this
          axis will create a new figure.
      - **sfile**: The file to save the pdf will the figures in
      - **ylim**: Optional. If None, the largest axis will be used for all figures
          else the value of this e.g. (0.8, 1) will be used for all figures.
      - **titles**: List of titles to put above the figures. Must have atleast
          same length as number of figures created.
    """
    if not kwargs_xdata:
        kwargs_xdata = {}
    if not kwargs_ydata:
        kwargs_ydata = {}
    if not kwargs_plot:
        kwargs_plot = {}

    record = records[record_name]
    xdata = record.select(**kwargs_xdata)
    ydata = record.select(**kwargs_ydata)
    # Buffer to store ylim. This is used to autmatically have
    # the same max y lim in all figures
    ylimset = [0, 0]


    figs = plot.multifig(xdata, ydata, fig_axis, kwargs_plot, titles)
    for fig in figs:
        ax = fig.get_axes()[0]
        plt.sca(ax)
        if ax.get_ylim()[0] < ylimset[0]:
            ylimset[0] = ax.get_ylim()[0]
        if ax.get_ylim()[1] > ylimset[1]:
            ylimset[1] = ax.get_ylim()[1]

    # Apply biggest ylim
    if not isinstance(ylim, type(None)):
        ylimset = ylim
    for fig in figs:
        for ax in fig.get_axes():
            ax.set_ylim(ylimset)

    if sfile:
        print('Saving multifig in ', os.path.abspath(sfile))
        sfgfig.save_figs_to_multipage_pdf(figs, sfile)
        for fig in figs:
            plt.close(fig)
        return
    return figs

def plot_contour(
    record_name,
    kwargs_data=None,
    kwargs_contourf=None,
    colorbar=True,
):
    if not kwargs_data:
        kwargs_data = {}
    if not kwargs_contourf:
        kwargs_contourf = {}

    kwargs_contourf.setdefault('levels', linspace(0.7, 1.1))
    kwargs_contourf.setdefault('extend', 'both')

    fig, ax = plt.subplots()
    record = records[record_name]
    x, y, z = record.contour(**kwargs_data)
    plot.contour(x, y, z, **kwargs_contourf )
    if colorbar:
        plt.colorbar()


def record_to_df(
        record,
        props=('pixel', 'wavenumber', 'wavelength', 'basesubed', 'normalized'),
        kwargs_select=None
):
    """Transform record to pandas DataFrame.

    This only works vor very simple records, with one spectrum and one pp-delay"""
    if not kwargs_select:
        kwargs_select = {}
    kwargs_select.setdefault('frame_med', True)
    data = [record.select(arg, **kwargs_select).squeeze() for arg in props]
    df = pd.DataFrame(
        data=data,
        index=props,
    )
    return df


def find(word):
    """Find the strign word in records keys and return the keys."""
    rk = list(records.keys())
    return list(array(rk)[[word in elm for elm in rk]])


def concatenate_list_of_SfgRecords(list_of_records):
    """Concatenate SfgRecords into one big SfgRecord."""

    concatable_attributes = ('rawData', 'base', 'norm')

    ret = SfgRecord()
    ret.metadata["central_wl"] = None
    # TODO Rewrite this pythonic
    for attribute in concatable_attributes:
        setattr(
            ret,
            attribute,
            concatenate(
                [getattr(elm, attribute) for elm in list_of_records],
                FRAME_AXIS_INDEX
            )
        )

    concatable_lists = ('wavelength', 'wavenumber')
    for attribute in concatable_lists:
        if all([all(getattr(elm, attribute)==getattr(list_of_records[0], attribute)) for elm in list_of_records]):
            setattr(ret, attribute, getattr(list_of_records[0], attribute))
            if attribute == 'wavenumber':
                ret._setted_wavenumber = True
            if attribute == 'wavelength':
                ret._setted_wavelength = True
        else:
            logger.debug('Not concatenating {}'.format(attribute))

    # Concatenate unlistable attributes
    concatable_attributes = (
        'het_shift', 'het_start', 'het_stop', 'pp_delays', 'norm_het_shift', 'norm_het_start', 'norm_het_stop',
    )
    for attribute in concatable_attributes:
        if all([getattr(elm, attribute) == getattr(list_of_records[0], attribute) for elm in list_of_records]):
            setattr(ret, attribute, getattr(list_of_records[0], attribute))

    # concat some properties
    ret.dates = concatenate([elm.dates for elm in list_of_records]).tolist()
    ret.pp_delays = list_of_records[0].pp_delays

    # Keep unchanged metadata and listify changed metadata.
    for key in list_of_records[0].metadata:
        values = [record.metadata.get(key) for record in list_of_records]
        if all([elm == values[0] for elm in values]):
            ret.metadata[key] = values[0]
        else:
            ret.metadata[key] = values

    return ret


def cache_records(records, cache_dir='./cache'):
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

# ## Aggregation Functions Templates
def get_basesubed(record, kwargs=None):
    """Get only the basesubed."""
    kwargs_select = {
        'prop': 'basesubed',
        'frame_med': True,
    }
    if isinstance(kwargs, dict):
        kwargs_select = {**kwargs_select, **kwargs}
    return record.select(**kwargs_select).flatten()


def get_normalized(record, kwargs=None):
    """Get only the basesubed."""
    kwargs_select = {
        'prop': 'normalized',
        'frame_med': True,
    }
    if isinstance(kwargs, dict):
        kwargs_select = {**kwargs_select, **kwargs}
    return record.select(**kwargs_select).flatten()


def get_wavenumbers(record, kwargs=None):
    kwargs_select = {
        'prop': 'wavenumber',
    }
    if isinstance(kwargs, dict):
        kwargs_select = {**kwargs_select, **kwargs}
    return record.select(**kwargs_select).flatten()


def get_sum(record, kwargs=None):
    """Get the sum of a record """

    kwargs_select = {
        'prop': 'normalized',
        'frame_med': True,
        'sum_pixel': True,
    }
    if isinstance(kwargs, dict):
        kwargs_select = {**kwargs_select, **kwargs}
    return float(record.select(**kwargs_select).flatten())


def get_sum_errors(record, kwargs=None):
    """Get uncertainties of the sums"""
    kwargs_select = {
        'prop': 'normalized',
        'frame_med': False,
        'sum_pixel': True,
    }
    if isinstance(kwargs, dict):
        kwargs_select = {**kwargs_select, **kwargs}
    return sem(record.select(**kwargs_select).flatten())
