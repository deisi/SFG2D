#!/usr/bin.env python
# coding: utf-8

"""Module for figure function."""

import os
import numpy as np
import matplotlib.pyplot as plt
import sfg2d

from .plot import fit_model

def ioff(func):
    """Decorator to make plotting non interactive temporally ."""
    def make_ioff(*args, **kwargs):
        plt.ioff()
        func(*args, **kwargs)
        plt.ion()

    return make_ioff


def save_figs_to_multipage_pdf(figs, fpath):
    """Save a list of figures into a multipage pdf.

    figs: list of figures to save to a multipage pdf.
    fpath: filepath of the pdf to save to.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if fpath[-4:] != '.pdf':
        fpath += '.pdf'

    with PdfPages(fpath) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    print("Saved figure to: {}".format(os.path.abspath(fpath)))


def multiplot(
        plots=None,
        kwargs_figure=None,
        args_subplot=[111],
        kwargs_subplot=None,
        setters_fig=None,
        setters_axis=None,
        legend=False,
        ticks_right=False,
        title=None,
        kwargs_subplots_adjust=None,
        xticks=None,
        yticks=None,
):
    if not kwargs_figure:
        kwargs_figure = {}
    if not kwargs_subplot:
        kwargs_subplot = {}
    fig = plt.figure(**kwargs_figure)
    ax = fig.add_subplot(*args_subplot, **kwargs_subplot)
    ax.cla()
    if kwargs_subplots_adjust:
        plt.subplots_adjust(**kwargs_subplots_adjust)
    if title:
        plt.title(title)
    if ticks_right:
        ax.yaxis.tick_right()

    if setters_fig:
        for setter_name, setter_value in setters_fig.items():
            setter_func = getattr(fig, setter_name)
            setter_func(**setter_value)

    for plot_config in plots:
        # name of the plot config is plot function name.
        # Therefore it must be stripped
        plot_func_name, plot_config = list(plot_config.items())[0]
        plot_func = getattr(sfg2d.plot, plot_func_name)

        record = plot_config['record']
        print('Select y data with: ', plot_config['kwargs_select_y'])
        ydata = record.select(**plot_config['kwargs_select_y'])
        print('Select x data with:', plot_config['kwargs_select_x'])
        xdata = record.select(**plot_config['kwargs_select_x'])
        kwargs_plot = plot_config.get('kwargs_plot', {})

        kwargs_select_yerr = plot_config.get('kwargs_select_yerr')
        if kwargs_select_yerr:
           yerr = record.sem(**kwargs_select_yerr)
           kwargs_plot['yerr'] = yerr

        plot_func(xdata, ydata, **kwargs_plot)

    if setters_axis:
        for setter_name, setter_value in setters_axis.items():
            setter_func = getattr(ax, setter_name)
            if isinstance(setter_value, dict):
                setter_func(**setter_value)
            else:
                setter_func(setter_value)

    if legend:
        if isinstance(legend, dict):
            ax.legend(**legend)
        else:
            ax.legend()

    if xticks:
        plt.xticks(xticks)
    if yticks:
        plt.yticks(yticks)
    return fig, ax

def spectrum(
        record,
        kwargs_subplot=None,
        kwargs_select=None,
        x_prop='range',
        x_prop_kw=None,
        save=False,
        title=None,
        fname=None,
        xlim=None,
        ylim=None,
        legend=None,
        **kwargs
):
    """Figure of Static spectrum from a record.

    record: Record to get data from
    save: Boolean, Save figure
    scale: Scale y axis.
    kwargs_select: dict passed to select method
    fig: give figure to plot on
    x_prop_kw: Dict for selection of x propertie
    ydata_prop: Allows to use a certain attribute of the ydata for the plot.

    Returns
      fig and ax.
    """
    if kwargs_subplot:
        fig, ax = plt.subplots(**kwargs_subplot)
        fig.clf()
    else:
        fig = plt.gcf()
        ax = plt.gca()
    # Must be None at first to prevent memory leackage from other calls
    if not x_prop_kw:
        x_prop_kw = {}
    if not kwargs_select:
        kwargs_select = {}

    kwargs_select.setdefault('delay_mean', True)
    kwargs_select.setdefault('frame_med', True)
    kwargs_select.setdefault('prop', 'unpumped')
    ydata = record.select(**kwargs_select)

    # Make sure that the use of roi_pixel doesn't fuck up figure axes
    kwargs_prop = kwargs_select.get('kwargs_prop')
    if kwargs_prop:
        roi_pixel = kwargs_prop.get('roi_pixel')
        if roi_pixel and x_prop in ('pixel', 'wavenumber', 'wavelength'):
            x_prop_kw['roi_pixel'] = roi_pixel

    # Handle xdata
    if x_prop == 'range':
        xdata = range(ydata.shape[-1])
    else:
        xdata = record.select(x_prop, **x_prop_kw)
        roi_pixel = kwargs_select.get('roi_pixel')
        if roi_pixel:
            xdata = xdata[roi_pixel]

    # Call of the plot function
    sfg2d.plot.spectrum(xdata, ydata, **kwargs)

    if not title:
        title = "{}".format(record.lname)
    plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if legend:
        if isinstance(legend, dict):
            plt.legend(**legend)
        else:
            plt.legend()
    if not fname:
        fname = 'figures/{}_static.pdf'.format(record.name)
    if save:
        print('Saving to: {}'.format(fname))
        plt.savefig(fname)
        print("saved")
    return fig, ax

def figure(
        record,
        select_ydata_kw,
        select_xdata_kw,
        plot_func,
        plot_func_attr,
        subplots_kw=None,
):
    if subplots_kw:
        fig, ax = plt.subplots(**subplots_kw)
    else:
        fig = plt.gcf()
        ax = plt.gca()

    xdata = record.select(**select_xdata_kw)
    ydata = record.select(**select_ydata_kw)

    plot_func = getattr(sfg2d.plot, plot_func)
    plot_func(xdata, ydata, **plot_func_attr)

    return fig, ax

def hot_and_cold(
        record_cold,
        record_hot,
        kwargs_subplot=None,
        kwargs_select_cold=None,
        kwargs_select_hot=None,
        x_prop='wavenumber',
        title=None,
        plot_hot_kw=None,
        plot_cold_kw=None,
        scale=1,
        fname='figures/hot_and_cold.pdf',
        save=False,
        legend=True,
        xlim=None,
        ylim=None,
):
    """Heat figure."""

    if not kwargs_subplot: kwargs_subplot = {}
    if not kwargs_select_cold: kwargs_select_cold = {}
    if not kwargs_select_hot: kwargs_select_hot = {}
    if not plot_hot_kw: plot_hot_kw = {}
    if not plot_cold_kw: plot_cold_kw = {}

    fig, ax = plt.subplots(**kwargs_subplot)
    fig.clf()


    for kwargs_select in (kwargs_select_hot, kwargs_select_cold):
        kwargs_select.setdefault('delay_mean', True)
        kwargs_select.setdefault('frame_med', True)
        kwargs_select.setdefault('prop', 'unpumped')
    cold = record_cold.select(**kwargs_select_cold)
    hot = record_hot.select(**kwargs_select_hot)
    plot_hot_kw.setdefault('label', 'Hot')
    plot_hot_kw.setdefault('color', 'C3')
    plot_cold_kw.setdefault('label', 'Cold')
    plot_cold_kw.setdefault('color', 'C0')
    sfg2d.plot.spectrum(record_hot.select(x_prop), scale*hot, **plot_hot_kw)
    sfg2d.plot.spectrum(record_cold.select(x_prop), scale*cold, **plot_cold_kw)

    if title:
        plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if save:
        print('Saving to: ', os.path.abspath(fname))
        plt.savefig(fname)
        print('DONE')

    if legend:
        plt.legend()
    return fig, ax


def heat_diff(
        record_cold,
        record_hot,
        opt='-',
        kwargs_subplot={},
        kwargs_select_cold={},
        kwargs_select_hot={},
        x_prop='wavenumber',
        title=None,
        plot_kw={},
        scale=1,
        fname='figures/heat_diff.pdf',
        save=False,
        legend=False,
        xlim=None,
        ylim=None,
):

    """Figure with the difference between cold and hot spectrum."""
    fig, ax = plt.subplots(**kwargs_subplot)
    fig.clf()

    for kwargs_select in (kwargs_select_hot, kwargs_select_cold):
        kwargs_select.setdefault('delay_mean', True)
        kwargs_select.setdefault('frame_med', True)
        kwargs_select.setdefault('prop', 'unpumped')
    cold = record_cold.select(**kwargs_select_cold)
    hot = record_hot.select(**kwargs_select_hot)

    if opt is '-':
        diff = hot-cold
    elif opt is '/':
        diff = hot/cold
    sfg2d.plot.spectrum(record_cold.select(x_prop), scale*diff, **plot_kw)

    if title:
        plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if save:
        print('Saving to: ', os.path.abspath(fname))
        plt.savefig(fname)
        print('DONE')

    if legend:
        plt.legend()
    return fig, ax


def pump_probe(
        record,
        kwargs_subplots={},
        kwargs_data={},
        kwargs_plot={},
        colorbar=True,
        title=None,
        xlim=None,
        ylim=None,
        xlabel='Time in fs',
        ylabel='Frequency in 1/cm',
        fname=None,
        savefig=False,
        close=False,
        skip=False,
):
    """Configurable contour plot.

    A contour plot of a record. By default it uses the relative
    bleach of  the record. At first a median filter of 7 pixels is used.
    Can be changed with: 'kwargs_data=dicht(medfilt_pixel=number)'. Afterwards
    an FFT based double resample filter with 30 Frequencies is used. Can
    be changed with 'kwargs_data=dict(resample_filter=number)'. To change the
    contrast of the plot, change the levels of the contour plot with:
    'kwargs_plot=dict(levels=arange(min, max, stepsize))'.

    **Arguments:**
      - **record**: The record to plot.
    **Keywords:**
      - **kwargs_subplots**: Keywords for subplot creation
      - **kwargs_data**: Keywords for record.contour data selection
      - **kwargs_plot**: Keywords for the contour plot.
      - **colorbar**: Boolean to show colorbar
      - **title**: Title string. By default tries to construct tile from record
      - **xlim**: X axis limit of the plot. Default is None
      - **ylim**: Y Axis limit of the plot. Default is None
      - **xalbel**: X label of the plot.
      - **ylabel**: Y label if the plot.
      - **fname**: File name to save figure with. If none given
           'figures/pump_probe.pdf' is used
      - **savefig**: Boolean to save figure
      - **close**: Boolean to close figure at the end.
      - **skip**: Boolean weather to skip the plot.

    **Returns**
    Figure object.
    """
    if skip:
        print("Skipping...")
        return

    fig, ax = plt.subplots(**kwargs_subplots)
    record.figures[fig.number] = fig

    kwargs_data.setdefault('resample_freqs', 30)
    kwargs_data.setdefault('medfilt_pixel', 7)
    x, y, z = record.contour(
        **kwargs_data
    )

    kwargs_plot.setdefault('extend', 'both')
    plt.contourf(x, y, z, **kwargs_plot)
    if colorbar:
        plt.colorbar()

    if not title:
        try:
            title = "{} Pump @ {}".format(record.lname, record.pump_freq)
        except AttributeError:
            title = ''
    plt.title(title)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not fname:
        fname = 'figures/pump_probe.pdf'

    if savefig:
        print('Saving to:', fname)
        plt.savefig(fname)
        print('DONE')

    if close:
        plt.close(fig)

    return fig


def trace(
        record,
        sl,
        kwargs_data={},
        fit_modekwargs_legend={},
        kwargs_subplots={},
        title=None,
        kwargs_plot={},
        errorbar_kwargs=None,
        xlim=None,
        ylim=None,
        xlabel='Time in fs',
        ylabel='Relative Bleach',
        legend=True,
        fname=None,
        save=False,
        close=False,
):
    """Figure of a trace.

    **Arguments:**
      - **record**: sfg2d.SfgRecord object
      - **sl**: slice that selects the region of interest in wavenumbers.

    **Optional fig_trace_config keywords:**
      - **kwargs_data**: Keywords of data selection. See `sfg2d.SfgRecord.trace`
        for more information.
      - **fit_modekwargs_legend**: Config of `fit_model`, See sfg2d.analyse.fit_model for
        further information.
      - **fig_kwargs**: Keywors of the subplots.
      - **title**: Title of the plot.
      - **errorbar_kwargs**: Keywords of the errorbar plot
      - **xlim**: Xlim of the plot
      - **ylim**: Ylim of the plot
      - **xlabel**: xlabel of the plot
      - **ylabel**: ylabel of the plot
      - **legend**: boolean it show legend. Defaults to true.
      - **fname**: String of filename to save figure at.
      - **save**: Boolean if figure should be saved to fname.

    """
    kwargs_data['roi_wavenumber'] =  sl
    x, y, yerr = record.trace(**kwargs_data)
    y, yerr = y.squeeze(), yerr.squeeze()

    # Indentifier is there for convenience
    _model_identifier = '' # Identifier for default filename if model is used.
    model_name = fit_modekwargs_legend.get('name')
    if model_name:
        _model_identifier = '_{}'.format({
            'FourLevelMolKinM': '4L',
            'SimpleDecay': 'SD'
        }.get(model_name, model_name))

    fig, ax = plt.subplots(**kwargs_subplots)

    if title:
        plt.title(title)

    if errorbar_kwargs:
        errorbar_kwargs.setdefault('marker', 'o')
        errorbar_kwargs.setdefault('label', 'Data')
        errorbar_kwargs.setdefault('linestyle', 'None')
        errorbar_kwargs.setdefault('axes', ax)
        plotline, capline, barline = plt.errorbar(
            x,
            y,
            yerr,
            **errorbar_kwargs
        )
    else:
        kwargs_plot.setdefault('marker', 'o')
        kwargs_plot.setdefault('linestyle', 'None')
        ax.plot(x, y, **kwargs_plot)

    model = None
    if model_name:
        model_key = '{}-{} {}'.format(
                sl.start, sl.stop, model_name
            )
        model = fit_model(x, y, yerr, **fit_modekwargs_legend)
        record.models[model_key] = model

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()

    if not fname:
        fname = 'figures/{}_pump{}_trace{}-{}{}.pdf'.format(
            record.name,
            record.pump_freq,
            sl.start,
            sl.stop,
            _model_identifier
        )

    if save:
        print('Saving to: ', os.path.abspath(fname))
        plt.savefig(fname)
        print('DONE')

    if close:
        plt.close(fig)

    return fig


def trace_model(
        model,
        kwargs_subplots={},
        title=None,
        errorbar_kwargs={},
        linekwargs_plot={},
        xlim=None,
        ylim=None,
        xlabel="Time in fs",
        ylabel="Bleach",
        fname=None,
        save=True,
        close=False,
        clf=True,
        ):
    """Figure of a model.

    **Arguments:**
      - **model**: The sfg2d.models.Model object to plot.
      - **config**: A dictionary with configuration parameters of the plot.

    **config**:
      - **fig_kwargs**: Dictionary to configure the figure with.
      - **ax**: Dictionary to configure the axes with
      - **title**: Title string of the figure
      - **error_kwargs**: Dictonary to configure the errorbar plot with.
    """
    fig, ax = plt.subplots(**kwargs_subplots)
    if clf:
        fig.clf()

    if title:
        plt.title(title)
    model.figures[fig.number] = fig

    errorbar_kwargs.setdefault('marker', 'o')
    errorbar_kwargs.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata, model.ydata, model.yerr, **errorbar_kwargs
    )

    linekwargs_plot.setdefault('color', plotline.get_color())
    plt.plot(model.xsample, model.yfit_sample, **linekwargs_plot)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not fname:
        fname = 'figures/trace_model.pdf'

    if save:
        print('Saving to: ', fname)
        plt.savefig(fname)
        print('DONE')

    if close:
        plt.close()


def models(
        models,
        models_kwargs_plot=[],
        num=None,
        title=None,
        fname=None
):
    """Plot list of given models.

    models: List of Models to plot
    models_kwargs_plot: List of configurations per plot
    """
    fig, ax = plt.subplots(num=num)
    fig.clf()
    for i in range(len(models)):
        model = models[i]
        kwargs_plot = {}
        try:
            kwargs_plot = models_kwargs_plot[i]
        except:
            pass
        sfg2d.plot.model(model, kwargs_plot)
    if title:
        plt.title(title)
    plt.legend()

    if fname:
        print('Saving to: ', fname)
        plt.savefig(fname)
        print('DONE')

    return fig, ax


def record_models(record, model_names, modekwargs_legend_plot=None):
    """Figure of multiple models from the same record.

    modekwargs_legend_plot: list of dicts. Each entry gets passed to
        `sfg2d.plot.model` as kwargs.
    """
    models = [record.models.get(model_name) for model_name in model_names]
    fig, ax = plt.subplots(
        num='{}_pump{}_traces'.format(record.name, record.pump_freq))
    fig.clf()
    if not modekwargs_legend_plot:
        modekwargs_legend_plot = [{} for model in models]
    for model, model_plot_kwg in zip(models, modekwargs_legend_plot):
        # Catch when model was not created
        if model:
            print(model_plot_kwg)
            sfg2d.plot.model(model, **model_plot_kwg)
    plt.legend()
    plt.xlabel('Time in fs')
    plt.ylabel('Relative Bleach')
    record.figures[fig.number] = fig


def bleach_slider(
        record,
        kwargs_selects={},
        x_prop='wavenumber',
        kwargs_plot={},
        scale=1,
        fig=None,
        ax=None,
        ylim=None,
        xlim=None,
        kwargs_legend={"loc": "lower left"},
):
    """Bleachplot, with slidable pp_delay index and autoscale.

    **Keywrords:**
      - **record**: The record to plot
      - **kwargs_select**: Select keywords to select data with.
         The default corresponds to:
         `{'prop': 'bleach', 'kwargs_prop':'{'prop':'basesubed'},
          'frame_med': True, 'medfilt_pixel':5}`
      - **scale**: Scaling factor for the data.
      - **fig**: figure
      - **ax**: axes
      - **ylim**: Optional tuple. Set fix ymin and ymax.
      - **xlim**: tuple to set xaxis.
      - **kwargs_legend**: Keywordsfor the plot legend
    """
    from ipywidgets import interact, widgets

    if not fig and not ax:
        fig, ax = plt.subplots()
    else:
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

    axes_lim_buffer = None

    kwargs_select_y = dict(**kwargs_selects)
    kwargs_select_y.setdefault('prop', 'bleach')
    kwargs_select_y.setdefault('kwargs_prop', {'prop': 'basesubed'})
    kwargs_select_y.setdefault('frame_med', True)
    kwargs_select_y.setdefault('medfilt_pixel', 5)

    @interact(
        Autoscale=True,
        index=widgets.IntSlider(
            max=record.number_of_pp_delays-1,
            continuous_update=False
        )
    )
    def my_plot(Autoscale, index):
        global axes_lim_buffer

        ax.clear()
        y = record.select(
            roi_delay=slice(index, index+1),
            **kwargs_select_y,
        )
        sfg2d.plot.spectrum(record.select(x_prop), scale*y, **kwargs_plot)

        if Autoscale:
            axes_lim_buffer = ax.get_xlim(), ax.get_ylim()

        if not isinstance(ylim, type(None)):
            ax.set_ylim(*ylim)
        elif not Autoscale:
            ax.set_ylim(axes_lim_buffer[1])

        if not isinstance(xlim, type(None)):
            ax.set_xlim(*xlim)
        elif not Autoscale:
            ax.set_xlim(axes_lim_buffer[0])

        ax.legend(**kwargs_legend)
        ax.figure.canvas.draw()

    return fig, ax


@ioff
def bleach_pdf(
        record,
        sfile,
        sfolder="./figures/",
        kwargs_selects={},
        kwargs_plot={},
        scale=1,
        xlim=None,
        ylim=None,
        x_prop='wavenumber',
        num_base='bl{}',
        xlabel='Wavenumber in 1/cm',
        ylabel=None,
        kwargs_legend={"loc": "lower left"},
        title_prefix=None,
        delay_offset=0,
):
    """Multipage pdf for the bleach plot.

    **Arguments**:
      - **record**: Record to plot data of
      - **sfile**: String with filename to save.
    **Keywords**:
      - **sfolder**: String with foldername to save file in.
      - **kwargs_selects**: Dict with keywords for selection of data.
        default corresponds to:
          {'prop': 'bleach', 'kwargs_prop':'{'prop':'basesubed'},
          'frame_med': True, 'medfilt_pixel':5}`
      - **kwargs_plot**: Keywords passed to the `plot_spce` function.
      - **scale**: Scaling factor for the data.
      - **ylim**: Optional tuple. Set fix ymin and ymax.
      - **xlim**: tuple to set xaxis.
      - **x_prop**: Propertie of the x axis
      - **num_base**: String to index the multiple plots with.
      - **xlabel**: String for the xlabel
      - **ylabel**: string for the y label
      - **kwargs_legend**: Keywordsfor the plot legend
      - **title_prefix**: Optinal String to prefix the title with.
           Default is record.metadata['material']
      - **delay_offset**: Offset to add to the delay.
    axes limits are always the same for all subplots.
    """

    kwargs_select_y = dict(**kwargs_selects)
    kwargs_select_y.setdefault('prop', 'bleach')
    kwargs_select_y.setdefault('kwargs_prop', {'prop': 'basesubed'})
    kwargs_select_y.setdefault('frame_med', True)
    kwargs_select_y.setdefault('medfilt_pixel', 5)
    figs = []
    for index in range(record.number_of_pp_delays):
        fig, ax = plt.subplots(num=num_base.format(index))
        figs.append(fig)

        y = record.select(
            roi_delay=slice(index, index+1),
            **kwargs_select_y,
        )
        x = record.select(prop=x_prop)
        sfg2d.plot.spectrum(x, scale*y, ax=ax, **kwargs_plot)

        if not title_prefix:
            title_prefix = record.metadata.get('material', '')
        ax.set_title("{} @ {} fs".format(
            title_prefix, record.pp_delays[index]+delay_offset)
        )
        ax.set_xlim(xlim)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # Prepare axes limits
    if isinstance(ylim, type(None)):
        ylim = [0, 0]
        for fig in figs:
            axes = fig.get_axes()
            for ax in axes:
                if ax.get_ylim()[0] < ylim[0]:
                    ylim[0] = ax.get_ylim()[0]
                if ax.get_ylim()[1] > ylim[1]:
                    ylim[1] = ax.get_ylim()[1]
    ymin, ymax = ylim

    for fig in figs:
        for ax in fig.get_axes():
            ax.set_ylim(ylim)

    save_figs_to_multipage_pdf(figs, sfolder+sfile)
    for fig in figs:
        plt.close(fig.number)


def spectrum_pump_vs_probe(
        record2d,
        delay,
        roi_pixel=slice(None),
        kwargs_pump_vs_probe={},
        kwargs_contour={},
        colorbar=True,
        diagonal=True,
        num=None,
        title='',
        tight_layout=False,
):
    """Figure of 2dRecprd."""

    fig, ax = plt.subplots(num=num)
    fig.clf()
    plt.title(title)
    kwargs_pump_vs_probe['delay'] = delay
    kwargs_pump_vs_probe['roi_pixel'] = roi_pixel
    x = record2d.pump_freqs
    y = record2d.wavenumbers[roi_pixel]
    z = record2d.pump_vs_probe(**kwargs_pump_vs_probe)
    plt.contourf(x, y, z, **kwargs_contour)
    plt.xlabel('Pump in 1/cm')
    plt.ylabel('Probe in 1/cm')
    if colorbar:
        plt.colorbar()
    if diagonal:
        l_min = np.max([x.min(), y.min()])
        l_max = np.min([x.max(), y.max()])
        plt.plot([l_min, l_max], [l_min, l_max], color='k')
    if tight_layout:
        plt.tight_layout()

    return fig


@ioff
def spectra_pump_vs_probe(
        record2d,
        roi_pump_freqs=slice(None),
        roi_pixel=slice(None),
        kwargs_pump_vs_probe={},
        kwargs_contour={},
        fig_name='',
        title='',
        close=True,
        save=True,
        colorbar=True,
        diagonal=True,
        tight_layout=False,
):
    """Saves pump vs probe spectra."""
    figures = []
    for delay in range(len(record2d.pp_delays)):
        fig, ax = plt.subplots(
            num=fig_name+"_pp_delay{:.0f}".format(record2d.pp_delays[delay])
        )
        figures.append(fig)
        fig.clf()
        plt.title(title + ' {:.0f} fs'.format(record2d.pp_delays[delay]))
        kwargs_pump_vs_probe['delay'] = delay
        kwargs_pump_vs_probe['roi_pixel'] = roi_pixel
        x = record2d.pump_freqs
        y = record2d.wavenumbers[roi_pixel]
        z = record2d.pump_vs_probe(**kwargs_pump_vs_probe)
        plt.contourf(x, y, z, **kwargs_contour)
        plt.xlabel('Pump in 1/cm')
        plt.ylabel('Probe in 1/cm')
        if colorbar:
            plt.colorbar()
        if diagonal:
            l_min = np.max([x.min(), y.min()])
            l_max = np.min([x.max(), y.max()])
            plt.plot([l_min, l_max], [l_min, l_max], color='k')
        if tight_layout:
            plt.tight_layout()

    if save:
        sfg2d.plotting.save_figs_to_multipage_pdf(
            figures,
            'figures/{}_record2d'.format(fig_name)
        )

    if close:
        for fig in figures:
            plt.close(fig)

    return figures


def spectra_static(
        record2d,
        kwargs_subplots={},
        kwargs_data={},
        title=None,
):
    """Static Spectra for measured Pump Frequencies."""
    fig, ax = plt.subplots(**kwargs_subplots)
    fig.clf()
    if title:
        plt.title(title)

    kwargs_data.setdefault('delay', 0)
    xdata = record2d.wavenumbers[kwargs_data.get('roi_pixel', slice(None))]
    ydata = record2d.static(**kwargs_data)
    plt.plot(xdata, ydata)
    plt.legend(['{:.0f} 1/cm'.format(elm) for elm in record2d.pump_freqs])
    plt.xlabel('Wavenumber in 1/cm')
    plt.ylabel('Normalized SFG in a.u.')
    return fig
