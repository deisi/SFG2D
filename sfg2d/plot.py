#!/usr/bin.env python
# coding: utf-8

"""Module for plot functions."""

import matplotlib.pyplot as plt
import sfg2d
from numpy import transpose

def fit_model(
        x,
        y,
        yerr,
        name,
        model_kwgs={},
        fit=True,
        print_matrix=True,
        plot_kwgs={},
        show_box=True,
        box_coords=None,
):
    """Fit Data in a configurable way.

    **Arguments:**
      - **x**: X data
      - **y**: Y data
      - **yerr**: Y error of the data
      - **name**: Name of the model

    **Keywords:**
      - **model_kwgs**: Keywords for the model
      - **fit**: Boolean to run fit
      - **print_matrix**: Boolean to print Correlation Matrix
      - **show_box**: Show fit result box
      - **box_coords**: Coordinates of the fit result box
    """
    model = getattr(sfg2d.models, name)(x, y, yerr, **model_kwgs)

    if fit:
        sfg2d.fit_model(
            model, print_matrix=print_matrix
        )
    plot_kwgs.setdefault('color', 'red')
    plot_kwgs.setdefault('label', 'Fit')
    plt.plot(model.xsample, model.yfit_sample, **plot_kwgs)

    if show_box:
        model.draw_text_box(box_coords)
    return model

def model(model, errorbar_kwgs=None, lineplot_kwgs=None):
    if not errorbar_kwgs:
        errorbar_kwgs = {}
    if not lineplot_kwgs:
        lineplot_kwgs = {}

    errorbar_kwgs.setdefault('marker', 'o')
    errorbar_kwgs.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata, model.ydata.T, model.yerr.T, **errorbar_kwgs
    )
    print(plotline.get_color())
    lineplot_kwgs.setdefault('color', plotline.get_color())
    print(lineplot_kwgs)
    plt.plot(model.xsample, model.yfit_sample.T, **lineplot_kwgs)

def points_modeled(x, y, yerr=None, xline=None, yline=None, point_kwgs={}, line_kwgs={}):
    """Plot points and line."""
    point_kwgs.setdefault('marker', 'o')
    point_kwgs.setdefault('linestyle', 'None')
    if  isinstance(yerr, type(None)):
        lines = plt.plot(x, y, **point_kwgs)
        point = lines[-1]
    else:
        point, capline, barline = plt.errorbar(
            x, y, yerr, **point_kwgs
           )

    if not isinstance(xline, type(None)) and not isinstance(yline, type(None)):
        # For some reason, color must be set explicitly here,
        # otherwise it is not respected. Its a workaround
        line_kwgs.setdefault('color', point.get_color())
        color = line_kwgs.pop('color')
        plt.plot(xline, yline, color=color, **line_kwgs)

def spectrum(
        xdata,
        ydata,
        *args,
        ax=None,
        xlabel='Wavenumber in 1/cm',
        ylabel='SFG Intensity in a.u.',
        **kwargs
):
    """
    Plot data with pixel axis of data as x-axis

    xdata: 1d numpy array for x-axis values.
    ydata: 4d numpy array with [delay, frame, spec, pixel]
    ax: axis obj to plot on
    xlabel: Label of the x-axis

    **kwargs are passed to ax.plot

    """
    if not ax:
        ax = plt.gca()

    for delay in ydata:
        for frame in delay:
            for spec in frame:
                ax.plot(xdata, spec.T, *args, **kwargs)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def track(
        ydata,
        *args,
        xdata=None,
        ax=None,
        xlabel="RunNumber",
        ylabel='SFG Intensity',
        **kwargs
):
    """A Track is a Time wise plot of the data.

    **Arguments:**
      - **ydata**: 4d Numpy array to create plot from

    **Keywords:**

    """
    if not ax:
        ax = plt.gca()
    delays, frames, spectra, pixel = ydata.shape
    if pixel != 1:
        raise IOError

    if delays != 1:
        raise IOError

    data = ydata[0, :, :, 0]
    if isinstance(xdata, type(None)):
        plt.plot(data, *args, **kwargs)
    else:
        plt.plot(xdata, data, *args, **kwargs)


def trace(
        xdata,
        ydata,
        ax=None,
        yerr=None,
        xlabel='Time in fs',
        ylabel='Bleach in a.u.',
        **kwargs
):
    """
    data is the result of a subselection.

    This plot has delays on its x-axis.

    yerr Error bar for the trace. Must have no frame dimension.
    if yerr is given frame dimension must be 1.
    """
    if not ax:
        ax = plt.gca()

    # Transpose because we want the delay axis to be the last axis
    # of the array.
    kwargs.setdefault('marker', 'o')

    y = ydata.T
    for i in range(len(y)):
        pixel = y[i]
        for j in range(len(pixel)):
            spec = pixel[j]
            if isinstance(yerr, type(None)):
                for frame in spec:
                    ax.plot(xdata, frame.T, **kwargs)
            else:
                plt.errorbar(xdata, spec[0], yerr[:, j, i], axes=ax, **kwargs)


