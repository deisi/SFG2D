"""Module to implement some plotting functions as I use them for pump-probe
Analysis. This is very high level and thus easy to use, but also very specific.

To use this module import it such that:
    `import sfg2d.ana.pumpProbePlots as pp`

And then overwite alteas the records dict with a dict of actual data.

    `pp.records = {a dict with some records}`

Optionaly overwrite the models with

    `pp.records = {a dict with some models}`

"""
import os
import sfg2d
import matplotlib.pyplot as plt
from numpy import linspace

# Dict of records.
records = {}
# Dicto of model.
models = {}

def plot_spectra(record_names, kwargs_xdata, kwargs_ydata, kwargs_plots=None):
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
        sfg2d.plot.spectrum(xdata, ydata, **kwargs_plot)

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
        sfg2d.plot.trace(xdata, ydata, yerr=yerr, **kwargs_plot)

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
        sfg2d.plot.track(ydata=ydata, **kwargs_plot)

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

@sfg2d.fig.ioff
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


    figs = sfg2d.plot.multifig(xdata, ydata, fig_axis, kwargs_plot, titles)
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
        sfg2d.fig.save_figs_to_multipage_pdf(figs, sfile)
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
    sfg2d.plot.contour(x, y, z, **kwargs_contourf )
    if colorbar:
        plt.colorbar()
