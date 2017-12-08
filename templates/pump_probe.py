#!/usr/bin.env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import sfg2d

# Contour Plot
def contour_plot(record, contour_plot_kwgs):
    """Configurable contour plot."""
    if contour_plot_kwgs.get('skip'):
        print("Skipping...")
        return
    data_kwgs = contour_plot_kwgs.get('data_kwgs', {})
    data_kwgs.setdefault('resample_freqs', 30)
    data_kwgs.setdefault('medfilt_pixel', 7)
    x, y, z = record.contour(
        **data_kwgs
    )

    fig_kwgs = contour_plot_kwgs.get('fig_kwgs', {})
    fig_kwgs.setdefault('figsize', (10, 6))
    fig_kwgs.setdefault('num', '{}_pump{}_contour'.format(
        record.name, record.pump_freq))
    fig, ax = plt.subplots(**fig_kwgs)
    fig.clf()

    plot_kwgs = contour_plot_kwgs.get('plot_kwgs', {})
    sfg2d.plotting.plot_contour(x, y, z, **plot_kwgs)

    title_str = contour_plot_kwgs.get('title', "{} Pump @ {}".format(
        record.lname, record.pump_freq))
    plt.title(title_str)

    plt.xlim(contour_plot_kwgs.get('xlim'))
    plt.ylim(contour_plot_kwgs.get('ylim'))
    plt.xlabel(contour_plot_kwgs.get('xlabel', 'Time in fs'))
    plt.ylabel(contour_plot_kwgs.get('ylabel', 'Frequency in 1/cm'))

    fname = contour_plot_kwgs.get(
        'fname', 'figures/' + fig_kwgs['num'] + '.pdf'
    )
    print('Saving to:', fname)
    if contour_plot_kwgs.get('savefig', False):
        plt.savefig(fname)
        print('DONE')
    if contour_plot_kwgs.get('close'):
        plt.close(fig)
    return x, y, z

# Fit Trace
def fit_trace(record, trace_config):
    """Run Single Trace fit."""
    if trace_config.get('skip'):
        return

    model_name=trace_config['model_name']
    sl = trace_config['sl']

    data_kwgs = trace_config.get('data_kwgs', {})
    data_kwgs.setdefault('roi_wavenumber', sl)
    x, y, yerr = record.trace(**data_kwgs)
    y, yerr = y.squeeze(), yerr.squeeze()

    model_kwgs = trace_config.get('model_kwgs', {})
    model = getattr(sfg2d.models, model_name)(x, y, yerr, **model_kwgs)
    if model_kwgs.get('roi'):
        yerr = yerr[model_kwgs.get('roi')]
    if trace_config.get('fit'):
        sfg2d.fit_model(
            model, print_matrix=trace_config.get('print_matrix', True)
        )

    fig_kwgs = trace_config.get('fig_kwgs', {})
    fig_kwgs.setdefault('num', '{}_pump{}_sl_{}-{}_{}'.format(
        record.name, record.pump_freq, sl.start, sl.stop, model_name
    ))
    fig, ax = plt.subplots(**fig_kwgs)
    fig.clf()

    title_str = trace_config.get('title', '{} Pump @{} {}-{}'.format(
        record.lname, record.pump_freq, sl.start, sl.stop)
    )
    plt.title(title_str)

    plot_kwgs = trace_config.get('plot_kwgs', {})
    errorbar_kwgs = trace_config.get('errorbar_kwgs', {})
    errorbar_kwgs.setdefault('marker', 'o')
    errorbar_kwgs.setdefault('label', 'Data')
    errorbar_kwgs.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata,
        model.ydata,
        yerr,
        **errorbar_kwgs
    )

    line_kwgs = trace_config.get('line_kwgs', {})
    line_kwgs.setdefault('color', 'red')
    plt.plot(model.xsample, model.yfit_sample, **line_kwgs)

    plt.xlim(trace_config.get('xlim'))
    plt.ylim(trace_config.get('ylim'))
    plt.xlabel(trace_config.get('xlabel', 'Time in fs'))
    plt.ylabel(trace_config.get('ylabel', 'Relative Bleach'))
    plt.legend()
    model.draw_text_box(trace_config.get('box_coords'))

    fname = trace_config.get(
        'fname',
        'figures/{}_pump{}_trace{}-{}_fit_{}.pdf'.format(
            record.name, record.pump_freq, sl.start, sl.stop, model_name
        )
    )
    print('Saving to: ', os.path.abspath(fname))
    if trace_config.get('save'):
        plt.savefig(fname)
        print('DONE')

    if trace_config.get('close'):
        plt.close(fig)

    return model


# Combination of models
def combine_models(config):
    """Combine given models in one plot.

    models: List of models.
    config:
    """
    fig_kwgs = config.get('fig_kwgs', {})
    fig, ax = plt.subplots(**fig_kwgs)
    fig.clf()

    title_str = config.get('title', "Combined Models")
    plt.title(title_str)

    models = config['models']
    for i in range(len(models)):
        model_config = models[i]
        model = model_config['model']
        errorbar_kwgs = model_config.get('errorbar_kwgs', {})
        errorbar_kwgs.setdefault('marker', 'o')
        errorbar_kwgs.setdefault('linestyle', 'None')
        plotline, capline, barline = plt.errorbar(
            model.xdata, model.ydata, model.yerr,
            **errorbar_kwgs
        )

        plot_kwgs = model_config.get('plot_kwgs', {})
        plot_kwgs.setdefault('color', plotline.get_color())
        plt.plot(model.xsample, model.yfit_sample, **plot_kwgs)
    plt.xlim(config.get('xlim'))
    plt.ylim(config.get('ylim'))
    plt.xlabel(config.get('xlabel', "Time in fs"))
    plt.ylabel(config.get('ylabel', "Counts"))
    plt.legend()
    plt.grid()

    fname = config.get('fname', 'figures/combined_models.pdf')
    print('Saving to: ', fname)
    if config.get('save'):
        plt.savefig(fname)
        print("DONE")
