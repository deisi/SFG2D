#!/usr/bin.env python
# coding: utf-8

"""Module for plot functions."""

import matplotlib.pyplot as plt
import sfg2d
from numpy import transpose, where, linspace, all, array

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
      - **kwargs_errorbar**: Kwargs passed to errorbar plot. Can be used to change e.g. color of the plot.
      - **kwargs_line_plot**: Kwargs passed to line plot of the fit line.
      - **shift_x**: Quick hack to shift the fit plot by x
      - **shift_y**: Qucick hack to shift the fit by y
      - **scale_y**: Qucik hack to scale fit by y
      - **normalize**: Normalize fit height to 1 - 0
      - **xsample_slice**: Tuple with x start and x stop for finding normalization minimun
      - **kwargs_textbox**: If kwargs textbox given, then a textbox with fitresults is drawn
    - **show_roi**: Mark the roi of the fit
    - **show_modeled_only**: Show only the fit, not the data.
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
               # We need a scatter like plot if its just one point
                if all(array(spec.shape) == 1):
                    kwargs.setdefault('marker', 'o')

                if isinstance(xdata, type(None)):
                    ax.plot(spec.T, *args, **kwargs)
                else:
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
        show_hlines=False,
        **kwargs
):
    """A Track is a Time wise plot of the data.

    **Arguments:**
      - **ydata**: 4d Numpy array to create plot from

    **Keywords:**
      - **show_vlines**: Boolean to show vertical lines after each scan.

    """
    if not ax:
        ax = plt.gca()

    delays, frames, spectra = ydata.shape
    # oder F, because ppelays is the first index and that
    # changes fastest
    ydata = ydata.reshape(delays*frames, spectra, order='F')

    for ispectrum in range(spectra):
        data = ydata[:, ispectrum]
        if isinstance(xdata, type(None)):
            ax.plot(data, **kwargs)
        else:
            ax.plot(xdata, data, **kwargs)
    if show_hlines:
        plt.vlines([delays*frame for frame in range(frames)], ydata.min(), ydata.max())


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
            for frame in spec:
                if isinstance(yerr, type(None)):
                    ax.plot(xdata, frame.T, **kwargs)
                else:
                    plt.errorbar(xdata, frame, yerr[:, j, i], axes=ax, **kwargs)

def contour(
        xdata,
        ydata,
        zdata,
        ax=None,
        **kwargs
):
    """
    Makes a contour plot for an SfgRecord.select() return for xdata, ydata and zdata.
    This makes multiple contour plots on top of each other. Normally more then one makes
    no sence but it works this way.
    **Arguments:**
      - **xdata**: Usually pp_delays
      - **ydata**: usually wavenumbers
      - **ydata**: usually bleach
    """
    kwargs.setdefault('extend', 'both')
    num_pp_delays, num_frames, num_spectra, num_pixel = zdata.shape
    for index_spectrum in range(num_spectra):
        for index_frame in range(num_frames):
            zzdata = zdata[:, index_frame, index_spectrum].T
            plt.contourf(xdata, ydata, zzdata, **kwargs)


def multifig(xdata, ydata, fig_axis=0, kwargs_plot=None, titles=None):
    """Create multiple figures for ydata, by using the axis of fig_ax.

    **Argument**:
      - **xdata**: 1D array usually wavenumbers
      - **ydata**: 4D array as usally.

    **kwargs**:
      - **fig_axis**: 0-3 and defines the axis of ydata that will be looped
        over during creation of the figures. Data is then taken from this
        axis per figure. 0  means 1 figure per pp_delay, 1 means 1 figure per
        frame and so on.
      - **kwargs_plot**: kwrges passed to spectrum plot.
      - **titles**: list of titles must have at least same length as number of figures.

    **Returns**:
    list of created figures.
    """
    if not kwargs_plot:
        kwargs_plot = {}

    fig_numbers = ydata.shape[fig_axis]
    figs, axs = [], []

    for i in range(fig_numbers):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)
        try:
            plt.title(titles[i])
        except TypeError:
            pass
        yd = ydata.take([i], fig_axis)
        spectrum(xdata, yd, **kwargs_plot)
    return figs
