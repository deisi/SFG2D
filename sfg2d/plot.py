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
        kwargs_model={},
        fit=True,
        print_matrix=True,
        kwargs_plot={},
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
      - **kwargs_model**: Keywords for the model
      - **fit**: Boolean to run fit
      - **print_matrix**: Boolean to print Correlation Matrix
      - **show_box**: Show fit result box
      - **box_coords**: Coordinates of the fit result box
    """
    model = getattr(sfg2d.models, name)(x, y, yerr, **kwargs_model)

    if fit:
        sfg2d.fit_model(
            model, print_matrix=print_matrix
        )
    kwargs_plot.setdefault('color', 'red')
    kwargs_plot.setdefault('label', 'Fit')
    plt.plot(model.xsample, model.yfit_sample, **kwargs_plot)

    if show_box:
        model.draw_text_box(box_coords)
    return model

def model(model, kwargs_errorbar=None, kwargs_line_plot=None):
    if not kwargs_errorbar:
        kwargs_errorbar = {}
    if not kwargs_line_plot:
        kwargs_line_plot = {}

    kwargs_errorbar.setdefault('marker', 'o')
    kwargs_errorbar.setdefault('linestyle', 'None')
    plotline, capline, barline = plt.errorbar(
        model.xdata, model.ydata.T, model.yerr.T, **kwargs_errorbar
    )
    print(plotline.get_color())
    kwargs_line_plot.setdefault('color', plotline.get_color())
    print(kwargs_line_plot)
    plt.plot(model.xsample, model.yfit_sample.T, **kwargs_line_plot)

def points_modeled(x, y, yerr=None, xline=None, yline=None, kwargs_point={}, kwargs_line={}):
    """Plot points and line."""
    kwargs_point.setdefault('marker', 'o')
    kwargs_point.setdefault('linestyle', 'None')
    if  isinstance(yerr, type(None)):
        lines = plt.plot(x, y, **kwargs_point)
        point = lines[-1]
    else:
        point, capline, barline = plt.errorbar(
            x, y, yerr, **kwargs_point
           )

    if not isinstance(xline, type(None)) and not isinstance(yline, type(None)):
        # For some reason, color must be set explicitly here,
        # otherwise it is not respected. Its a workaround
        kwargs_line.setdefault('color', point.get_color())
        color = kwargs_line.pop('color')
        plt.plot(xline, yline, color=color, **kwargs_line)

def spectrum(
        xdata,
        ydata,
        *args,
        ax=None,
        xlabel=None,
        ylabel=None,
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


