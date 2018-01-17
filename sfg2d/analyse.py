
#!/usr/bin.env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import sfg2d


### Deprecated

# Contour Plot
def fig_pump_probe(record, fig_pump_probe_kwgs):
    """Configurable contour plot."""
    if fig_pump_probe_kwgs.get('skip'):
        print("Skipping...")
        return

    fig_kwgs = fig_pump_probe_kwgs.get('fig_kwgs', None)
    if isinstance(fig_kwgs, type(None)):
        fig = plt.gcf()
    else:
        fig_kwgs.setdefault('figsize', (10, 6))
        fig_kwgs.setdefault('num', '{}_pump{}_contour'.format(
            record.name, record.pump_freq))
        fig = plt.figure(**fig_kwgs)
        fig.clf()
    if fig_pump_probe_kwgs.get('ax'):
        ax = plt.axes(**fig_pump_probe_kwgs['ax'])
    else:
        ax = plt.gca()
    record.figures[fig.number] = fig

    data_kwgs = fig_pump_probe_kwgs.get('data_kwgs', {})
    data_kwgs.setdefault('resample_freqs', 30)
    data_kwgs.setdefault('medfilt_pixel', 7)
    x, y, z = record.contour(
        **data_kwgs
    )

    plot_kwgs = fig_pump_probe_kwgs.get('plot_kwgs', {})
    sfg2d.plotting.plot_contour(x, y, z, **plot_kwgs)

    title_str = fig_pump_probe_kwgs.get('title', "{} Pump @ {}".format(
        record.lname, record.pump_freq))
    plt.title(title_str)

    plt.xlim(fig_pump_probe_kwgs.get('xlim'))
    plt.ylim(fig_pump_probe_kwgs.get('ylim'))
    plt.xlabel(fig_pump_probe_kwgs.get('xlabel', 'Time in fs'))
    plt.ylabel(fig_pump_probe_kwgs.get('ylabel', 'Frequency in 1/cm'))

    fname = fig_pump_probe_kwgs.get(
        'fname', 'figures/' + fig_kwgs['num'] + '.pdf'
    )
    print('Saving to:', fname)
    if fig_pump_probe_kwgs.get('savefig', False):
        plt.savefig(fname)
        print('DONE')
    if fig_pump_probe_kwgs.get('close'):
        plt.close(fig)
    return x, y, z

def fit_model(x, y, yerr, config):
    """Fit Data in a configurable way.

    **Arguments:**
    **x**: X data
    **y**: Y data
    **yerr**: Y error of the data
    **config**: Configuration dict.
      must contain at least a name of the model to use.

    """
    name = config['name']
    model_kwgs = config.get('model_kwgs', {})
    model = getattr(sfg2d.models, name)(x, y, yerr, **model_kwgs)

    if config.get('fit', True):
        sfg2d.fit_model(
            model, print_matrix=config.get('print_matrix', True)
        )
    plot_kwgs = config.get('plot_kwgs', {})
    plot_kwgs.setdefault('color', 'red')
    plot_kwgs.setdefault('label', 'Fit')
    plt.plot(model.xsample, model.yfit_sample, **plot_kwgs)

    if config.get('show_box', True):
        model.draw_text_box(config.get('box_coords'))
    return model


def trace(record, trace_config):
    """Plot Trace.

    **Arguments:**
      - **record**: sfg2d.SfgRecord object
      - **trace_config**: Config dict of the plot.
      Must atleas have a `sl` keyword, to slice the record with.
      Minimal example is:

    **Optional trace_config keywords:**
      - **data_kwgs**: Keywords of data selection. See `sfg2d.SfgRecord.trace`
        for more information.
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
      - **model**: Config of `fit_model`, See sfg2d.analyse.fit_model for
        further information.

    """
    sl = trace_config['sl']
    data_kwgs = trace_config.get('data_kwgs', {})
    data_kwgs.setdefault('roi_wavenumber', sl)
    x, y, yerr = record.trace(**data_kwgs)
    y, yerr = y.squeeze(), yerr.squeeze()

    _model_identifier = '' # Identifier for default filename if model is used.
    model_config = trace_config.get('model')
    if model_config:
        _model_identifier = '_{}'.format({
            'FourLevelMolKinM': '4L',
            'SimpleDecay': 'SD'
        }.get(model_config['name'], model_config['name']))
    fig_kwgs = trace_config.get('fig_kwgs')
    if isinstance(fig_kwgs, type(None)):
        fig = plt.gcf()
    else:
        fig_kwgs.setdefault(
            'num', '{}_pump{}_{}-{}{}'.format(
                record.name,
                record.pump_freq,
                sl.start,
                sl.stop,
                _model_identifier,
            ))
        fig = plt.figure(**fig_kwgs)
        fig.clf()
    if trace_config.get('ax'):
        ax = plt.axes(**trace_config['ax'])
    else:
        ax = plt.gca()
    record.figures[fig.number] = fig

    title_str = trace_config.get('title', '{} Pump @{} {}-{}'.format(
        record.lname, record.pump_freq, sl.start, sl.stop)
    )
    plt.title(title_str)

    errorbar_kwgs = trace_config.get('errorbar_kwgs', {})
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

    model = None
    if model_config:
        model_name = trace_config.get(
            'model_name', '{}-{} {}'.format(
                sl.start, sl.stop, model_config['name']
            )
        )
        model = fit_model(x, y, yerr, model_config)
        record.models[model_name] = model

    plt.xlim(trace_config.get('xlim'))
    plt.ylim(trace_config.get('ylim'))
    plt.xlabel(trace_config.get('xlabel', 'Time in fs'))
    plt.ylabel(trace_config.get('ylabel', 'Relative Bleach'))
    if trace_config.get('legend', True):
        plt.legend()

    fname = trace_config.get(
        'fname',
        'figures/{}_pump{}_trace{}-{}{}.pdf'.format(
            record.name, record.pump_freq, sl.start, sl.stop, _model_identifier
        )
    )
    print('Saving to: ', os.path.abspath(fname))
    if trace_config.get('save'):
        plt.savefig(fname)
        print('DONE')

    if trace_config.get('close'):
        plt.close(fig)

    return x, y, yerr, model


def plot_model(model, config):
    """Plot model."""
    fig_kwgs = config.get('fig_kwgs', None)
    if isinstance(fig_kwgs, type(None)):
        fig = plt.gcf()
    else:
        fig = plt.figure(**fig_kwgs)
        fig.clf()
    if config.get('ax'):
        ax = plt.axes(**config['ax'])
    else:
        ax = plt.gca()

    if config.get('title'):
        plt.title(config.get('title'))
    model.figures[fig.number] = fig

    errorbar_kwgs = config.get('errorbar_kwgs', {})
    errorbar_kwgs.setdefault('marker', 'o')
    errorbar_kwgs.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata, model.ydata, model.yerr, **errorbar_kwgs
    )
    lineplot_kwgs = config.get('lineplot_kwgs', {})
    lineplot_kwgs.setdefault('color', plotline.get_color())
    plt.plot(model.xsample, model.yfit_sample, **lineplot_kwgs)

    plt.xlim(config.get('xlim'))
    plt.ylim(config.get('ylim'))
    plt.xlabel(config.get('xlabel', "Time in fs"))
    plt.ylabel(config.get('ylabel', "Bleach"))

    fname = config.get('fname', 'figures/model.pdf')
    if config.get('save', False):
        print('Saving to: ', fname)
        plt.savefig(fname)
        print('DONE')


def plot_models(models, models_plot_kwgs=[], num=None, title=None, fname=None):
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
        plot_model(model, plot_kwgs)
    if title:
        plt.title(title)
    plt.legend()

    if fname:
        print('Saving to: ', fname)
        plt.savefig(fname)
        print('DONE')

    return fig, ax
