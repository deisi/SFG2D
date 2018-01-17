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


def spectrum(
        record,
        subplot_kw={},
        scale=1,
        select_kw={},
        x_prop='wavenumber',
        save=False,
        title=None,
        fname=None,
        xlim=None,
        ylim=None,
        **kwargs
):
    """Figure of Static spectrum from a record.

    record: Record to get data from
    save: Boolean, Save figure
    scale: Scale y axis.
    select_kw: dict passed to select method

    Returns
      fig and ax.
    """
    subplot_kw.setdefault('num', '{}_static'.format(record.name))
    fig, ax = plt.subplots(**subplot_kw)
    fig.clf()
    select_kw.setdefault('delay_mean', True)
    select_kw.setdefault('frame_med', True)
    select_kw.setdefault('prop', 'unpumped')
    data = record.select(**select_kw)
    sfg2d.plot.spectrum(record.select(x_prop), scale*data, **kwargs)
    if not title:
        title = "{}".format(record.lname)
    plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if not fname:
        fname = 'figures/{}_static.pdf'.format(record.name)
    if save:
        print('Saving to: {}'.format(fname))
        plt.savefig(fname)
        print("saved")
    return fig, ax


def hot_and_cold(
        record_cold,
        record_hot,
        subplot_kw={},
        select_kw_cold={},
        select_kw_hot={},
        x_prop='wavenumber',
        title=None,
        plot_hot_kw={},
        plot_cold_kw={},
        scale=1,
        fname='figures/hot_and_cold.pdf',
        save=False,
        legend=True,
        xlim=None,
        ylim=None,
):
    """Heat figure."""
    fig, ax = plt.subplots(**subplot_kw)
    fig.clf()

    for select_kw in (select_kw_hot, select_kw_cold):
        select_kw.setdefault('delay_mean', True)
        select_kw.setdefault('frame_med', True)
        select_kw.setdefault('prop', 'unpumped')
    cold = record_cold.select(**select_kw_cold)
    hot = record_hot.select(**select_kw_hot)
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
        subplot_kw={},
        select_kw_cold={},
        select_kw_hot={},
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
    fig, ax = plt.subplots(**subplot_kw)
    fig.clf()

    for select_kw in (select_kw_hot, select_kw_cold):
        select_kw.setdefault('delay_mean', True)
        select_kw.setdefault('frame_med', True)
        select_kw.setdefault('prop', 'unpumped')
    cold = record_cold.select(**select_kw_cold)
    hot = record_hot.select(**select_kw_hot)

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
        subplot_kwgs={},
        data_kwgs={},
        plot_kwgs={},
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
    Can be changed with: 'data_kwgs=dicht(medfilt_pixel=number)'. Afterwards
    an FFT based double resample filter with 30 Frequencies is used. Can
    be changed with 'data_kwgs=dict(resample_filter=number)'. To change the
    contrast of the plot, change the levels of the contour plot with:
    'plot_kwgs=dict(levels=arange(min, max, stepsize))'.

    **Arguments:**
      - **record**: The record to plot.
    **Keywords:**
      - **subplot_kwgs**: Keywords for subplot creation
      - **data_kwgs**: Keywords for record.contour data selection
      - **plot_kwgs**: Keywords for the contour plot.
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

    fig, ax = plt.subplots(**subplot_kwgs)
    record.figures[fig.number] = fig

    data_kwgs.setdefault('resample_freqs', 30)
    data_kwgs.setdefault('medfilt_pixel', 7)
    x, y, z = record.contour(
        **data_kwgs
    )

    plot_kwgs.setdefault('extend', 'both')
    plt.contourf(x, y, z, **plot_kwgs)
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
        data_kwgs={},
        fit_model_kwgs={},
        subplot_kwgs={},
        title=None,
        plot_kwgs={},
        errorbar_kwgs=None,
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
      - **data_kwgs**: Keywords of data selection. See `sfg2d.SfgRecord.trace`
        for more information.
      - **fit_model_kwgs**: Config of `fit_model`, See sfg2d.analyse.fit_model for
        further information.
      - **fig_kwgs**: Keywors of the subplots.
      - **title**: Title of the plot.
      - **errorbar_kwgs**: Keywords of the errorbar plot
      - **xlim**: Xlim of the plot
      - **ylim**: Ylim of the plot
      - **xlabel**: xlabel of the plot
      - **ylabel**: ylabel of the plot
      - **legend**: boolean it show legend. Defaults to true.
      - **fname**: String of filename to save figure at.
      - **save**: Boolean if figure should be saved to fname.

    """
    data_kwgs['roi_wavenumber'] =  sl
    x, y, yerr = record.trace(**data_kwgs)
    y, yerr = y.squeeze(), yerr.squeeze()

    # Indentifier is there for convenience
    _model_identifier = '' # Identifier for default filename if model is used.
    model_name = fit_model_kwgs.get('name')
    if model_name:
        _model_identifier = '_{}'.format({
            'FourLevelMolKinM': '4L',
            'SimpleDecay': 'SD'
        }.get(model_name, model_name))

    fig, ax = plt.subplots(**subplot_kwgs)

    if title:
        plt.title(title)

    if errorbar_kwgs:
        errorbar_kwgs.setdefault('marker', 'o')
        errorbar_kwgs.setdefault('label', 'Data')
        errorbar_kwgs.setdefault('linestyle', 'None')
        errorbar_kwgs.setdefault('axes', ax)
        plotline, capline, barline = plt.errorbar(
            x,
            y,
            yerr,
            **errorbar_kwgs
        )
    else:
        plot_kwgs.setdefault('marker', 'o')
        plot_kwgs.setdefault('linestyle', 'None')
        ax.plot(x, y, **plot_kwgs)

    model = None
    if model_name:
        model_key = '{}-{} {}'.format(
                sl.start, sl.stop, model_name
            )
        model = fit_model(x, y, yerr, **fit_model_kwgs)
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
        subplot_kwgs={},
        title=None,
        errorbar_kwgs={},
        lineplot_kwgs={},
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
      - **fig_kwgs**: Dictionary to configure the figure with.
      - **ax**: Dictionary to configure the axes with
      - **title**: Title string of the figure
      - **error_kwgs**: Dictonary to configure the errorbar plot with.
    """
    fig, ax = plt.subplots(**subplot_kwgs)
    if clf:
        fig.clf()

    if title:
        plt.title(title)
    model.figures[fig.number] = fig

    errorbar_kwgs.setdefault('marker', 'o')
    errorbar_kwgs.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata, model.ydata, model.yerr, **errorbar_kwgs
    )

    lineplot_kwgs.setdefault('color', plotline.get_color())
    plt.plot(model.xsample, model.yfit_sample, **lineplot_kwgs)

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
        models_plot_kwgs=[],
        num=None,
        title=None,
        fname=None
):
    """Plot list of given models.

    models: List of Models to plot
    models_plot_kwgs: List of configurations per plot
    """
    fig, ax = plt.subplots(num=num)
    fig.clf()
    for i in range(len(models)):
        model = models[i]
        plot_kwgs = {}
        try:
            plot_kwgs = models_plot_kwgs[i]
        except:
            pass
        sfg2d.plot.model(model, plot_kwgs)
    if title:
        plt.title(title)
    plt.legend()

    if fname:
        print('Saving to: ', fname)
        plt.savefig(fname)
        print('DONE')

    return fig, ax


def record_models(record, model_names, model_plot_kwgs=None):
    """Figure of multiple models from the same record.

    model_plot_kwgs: list of dicts. Each entry gets passed to
        `sfg2d.plot.model` as kwargs.
    """
    models = [record.models.get(model_name) for model_name in model_names]
    fig, ax = plt.subplots(
        num='{}_pump{}_traces'.format(record.name, record.pump_freq))
    fig.clf()
    if not model_plot_kwgs:
        model_plot_kwgs = [{} for model in models]
    for model, model_plot_kwg in zip(models, model_plot_kwgs):
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
        select_kws={},
        x_prop='wavenumber',
        plot_kwgs={},
        scale=1,
        fig=None,
        ax=None,
        ylim=None,
        xlim=None,
        l_kwgs={"loc": "lower left"},
):
    """Bleachplot, with slidable pp_delay index and autoscale.

    **Keywrords:**
      - **record**: The record to plot
      - **select_kw**: Select keywords to select data with.
         The default corresponds to:
         `{'prop': 'bleach', 'prop_kwgs':'{'prop':'basesubed'},
          'frame_med': True, 'medfilt_pixel':5}`
      - **scale**: Scaling factor for the data.
      - **fig**: figure
      - **ax**: axes
      - **ylim**: Optional tuple. Set fix ymin and ymax.
      - **xlim**: tuple to set xaxis.
      - **l_kwgs**: Keywordsfor the plot legend
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

    select_y_kws = dict(**select_kws)
    select_y_kws.setdefault('prop', 'bleach')
    select_y_kws.setdefault('prop_kwgs', {'prop': 'basesubed'})
    select_y_kws.setdefault('frame_med', True)
    select_y_kws.setdefault('medfilt_pixel', 5)

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
            **select_y_kws,
        )
        sfg2d.plot.spectrum(record.select(x_prop), scale*y, **plot_kwgs)

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

        ax.legend(**l_kwgs)
        ax.figure.canvas.draw()

    return fig, ax


@ioff
def bleach_pdf(
        record,
        sfile,
        sfolder="./figures/",
        select_kws={},
        plot_kwgs={},
        scale=1,
        xlim=None,
        ylim=None,
        x_prop='wavenumber',
        num_base='bl{}',
        xlabel='Wavenumber in 1/cm',
        ylabel=None,
        l_kwgs={"loc": "lower left"},
        title_prefix=None,
        delay_offset=0,
):
    """Multipage pdf for the bleach plot.

    **Arguments**:
      - **record**: Record to plot data of
      - **sfile**: String with filename to save.
    **Keywords**:
      - **sfolder**: String with foldername to save file in.
      - **select_kws**: Dict with keywords for selection of data.
        default corresponds to:
          {'prop': 'bleach', 'prop_kwgs':'{'prop':'basesubed'},
          'frame_med': True, 'medfilt_pixel':5}`
      - **plot_kwgs**: Keywords passed to the `plot_spce` function.
      - **scale**: Scaling factor for the data.
      - **ylim**: Optional tuple. Set fix ymin and ymax.
      - **xlim**: tuple to set xaxis.
      - **x_prop**: Propertie of the x axis
      - **num_base**: String to index the multiple plots with.
      - **xlabel**: String for the xlabel
      - **ylabel**: string for the y label
      - **l_kwgs**: Keywordsfor the plot legend
      - **title_prefix**: Optinal String to prefix the title with.
           Default is record.metadata['material']
      - **delay_offset**: Offset to add to the delay.
    axes limits are always the same for all subplots.
    """

    select_y_kws = dict(**select_kws)
    select_y_kws.setdefault('prop', 'bleach')
    select_y_kws.setdefault('prop_kwgs', {'prop': 'basesubed'})
    select_y_kws.setdefault('frame_med', True)
    select_y_kws.setdefault('medfilt_pixel', 5)
    figs = []
    for index in range(record.number_of_pp_delays):
        fig, ax = plt.subplots(num=num_base.format(index))
        figs.append(fig)

        y = record.select(
            roi_delay=slice(index, index+1),
            **select_y_kws,
        )
        x = record.select(prop=x_prop)
        sfg2d.plot.spectrum(x, scale*y, ax=ax, **plot_kwgs)

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
        pump_vs_probe_kwgs={},
        contour_kwgs={},
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
    pump_vs_probe_kwgs['delay'] = delay
    pump_vs_probe_kwgs['roi_pixel'] = roi_pixel
    x = record2d.pump_freqs
    y = record2d.wavenumbers[roi_pixel]
    z = record2d.pump_vs_probe(**pump_vs_probe_kwgs)
    plt.contourf(x, y, z, **contour_kwgs)
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
        pump_vs_probe_kwgs={},
        contour_kwgs={},
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
        pump_vs_probe_kwgs['delay'] = delay
        pump_vs_probe_kwgs['roi_pixel'] = roi_pixel
        x = record2d.pump_freqs
        y = record2d.wavenumbers[roi_pixel]
        z = record2d.pump_vs_probe(**pump_vs_probe_kwgs)
        plt.contourf(x, y, z, **contour_kwgs)
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
        subplot_kwgs={},
        data_kwgs={},
        title=None,
):
    """Static Spectra for measured Pump Frequencies."""
    fig, ax = plt.subplots(**subplot_kwgs)
    fig.clf()
    if title:
        plt.title(title)

    data_kwgs.setdefault('delay', 0)
    xdata = record2d.wavenumbers[data_kwgs.get('roi_pixel', slice(None))]
    ydata = record2d.static(**data_kwgs)
    plt.plot(xdata, ydata)
    plt.legend(['{:.0f} 1/cm'.format(elm) for elm in record2d.pump_freqs])
    plt.xlabel('Wavenumber in 1/cm')
    plt.ylabel('Normalized SFG in a.u.')
    return fig
