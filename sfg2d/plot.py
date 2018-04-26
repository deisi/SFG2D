#!/usr/bin.env python
# coding: utf-8

"""Module for plot functions."""

import matplotlib.pyplot as plt
import sfg2d
from numpy import transpose, where, linspace

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


def model_plot(
        model,
        kwargs_errorbar=None,
        kwargs_line_plot=None,
        shift_x=None,
        shift_y=None,
        scale_y=None,
        normalize=False,
        xsample_slice=(0, 1),
        kwargs_textbox=None,
        show_roi=True,
        show_modeled_only=True,
):
    """
    **Kwargs:**
      - **xsample_slice**: Tuple with x start and x stop for finding normalization minimun
      - **kwargs_textbox**: If kwargs textbox given, then a textbox with fitresults is drawn
    """
    if not kwargs_errorbar:
        kwargs_errorbar = {}
    if not kwargs_line_plot:
        kwargs_line_plot = {}

    kwargs_errorbar.setdefault('marker', 'o')
    kwargs_errorbar.setdefault('linestyle', 'None')

    if show_modeled_only:
        xdata = model.xdata
        ydata = model.ydata
        yerr = model.sigma
    else:
        xdata = model._xdata
        ydata = model._ydata
        yerr = model._sigma

    xsample = linspace(xdata[0], xdata[-1], model._xsample_num)
    ysample = model.fit_res(xsample)

    if shift_x:
        xdata =  xdata + shift_x
        xsample = xsample + shift_x

    if scale_y:
        ydata = ydata * scale_y
        ysample = ysample * scale_y
        yerr = yerr * scale_y

    if normalize:
        x_mask = where((xsample > xsample_slice[0]) & (xsample < xsample_slice[1]))
        factor = 1 - ysample[x_mask].min()
        ydata = (ydata - 1) / factor + 1
        ysample = (ysample - 1) / factor + 1
        yerr = yerr / factor

    if shift_y:
        ydata = ydata + shift_y
        ysample = ysample + shift_y


    plotline, capline, barline = plt.errorbar(
        xdata, ydata, yerr, **kwargs_errorbar
    )
    kwargs_line_plot.setdefault('color', plotline.get_color())
    plt.plot(xsample, ysample, **kwargs_line_plot)

    if show_roi:
        xx = xdata[model.roi]
        yy = ydata[model.roi]
        x = xx[0], xx[-1]
        y = yy[0], yy[-1]
        plt.scatter(x, y, marker='x', color='r')

    if isinstance(kwargs_textbox, dict):
        fig = plt.gcf()
        kwargs_textbox.setdefault('x', .6)
        kwargs_textbox.setdefault('y', .12)
        kwargs_textbox.setdefault('s', model.box_str)
        kwargs_textbox.setdefault('transform', fig.transFigure)
        plt.text(**kwargs_textbox)

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
        xdata=None,
        ydata=None,
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

    print(ydata.shape)
    delays, frames, spectra = ydata.shape
    ydata = ydata.reshape(delays*frames, spectra)

    for ispectrum in range(spectra):
        data = ydata[:, ispectrum]
        if isinstance(xdata, type(None)):
            plt.plot(data, **kwargs)
        else:
            plt.plot(xdata, data, **kwargs)


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


