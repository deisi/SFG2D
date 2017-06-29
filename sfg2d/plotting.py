from os import path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .utils import get_interval_index

def multipage_pdf(plot_func):
    """
    Function can be used as a decorator, to loop over a given range
    and call a plot function with an index parameter in each loop
    incerement. E.g.

    ```
    @sfg2d.multipage_pdf
    def my_plot(index, record):
        record.plot_bleach(attribute="bleach", x_axis="wavenumber", pp_delays=[index]);
        title(r"Sample @ %i fs"%record.pp_delays[index])
        xlim(2100, 2800)
        ylim(-0.003, 0.002)

    somesfgrecord = SfgRecord()
    my_plot("delme", range(2), somesfgrecord)
    ```


    plot_func: function
        ploting function with index value as and pssibly further args
        and kwargs as parameter.
        each time the index is increased and plot_func is called
        with the new index again.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    def make_multipage_pdf(name, inds, *args, **kwargs):
        """
        fname: string
            filename to save the multipage pdf in.
        indes: iterable
            iterable to loop over
        args and kwargs get passed to the plot function you use this
        decorator on.
        """
        if name[-4:] != '.pdf':
            name += '.pdf'
        with PdfPages(name) as pdf:
            for index in inds:
                fig, ax = plt.subplots()
                plot_func(index, *args, **kwargs)
                pdf.savefig()
                plt.close()

        print("Saved figure to: {}".format(path.abspath(name)))
    return make_multipage_pdf

def plot(*args, **kwargs):
    """Wrapper for default matplotlib.plot that adds xlabel as wavenumbers"""
    plt.plot(*args, **kwargs)
    plt.xlabel('Wavenumbers in cm$^{-1}$')


def plot_time(time, data, **kwargs):
    """ Wrapper function to plot formatted time on the x-axis
    and data on the y axis. If time is datetime obj, the time is
    plotted in HH:MM, if time is timedelta it is plotted as minuits

    Parameters
    ----------
    time: list of datetime or timedelta
        The times for the x-axis
    data: array like
        The data for the Y axis
    """
    import datetime
    from matplotlib.dates import DateFormatter

    fig = plt.gcf()
    ax = plt.gca()
    if isinstance(time[0], datetime.datetime):
        fig.autofmt_xdate()
        xfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        plt.xlabel("Time")
    if isinstance(time[0], datetime.timedelta):
        time = [elm.seconds//60 for elm in time]
        plt.xlabel('Minutes')
        plt.xlabel("Time in min")

    l0 = plt.plot(time, data, **kwargs)
    plt.plot(time, data, 'o', color=l0[0].get_color())

    # Append 5% margin left and right to the plot,
    # so that it looks nicer
    x_range = max(time) - min(time)
    x_range *= 0.05
    plt.xlim(min(time) - x_range, max(time) + x_range)

def errorshadow(x, y, dy, ax=None, **kwargs):
    """
    Plot error-bar as shadow around the data

    Parameters
    ----------
    x : array
        x-data
    y : array
        y-data
    dy : array
        uncertainty of the y data
    ax : Optional [matplotlib.axes obj]
        the aces to plot on to.
    **kwargs :
        kwargs are passed to `matplotlib.plot`
    """
    if not ax:
        ax = plt.gca()
    lines = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y-dy, y+dy, color=lines[0].get_color(), alpha=0.5)

def contour(x, y, z, N=30, fig=None,
            y_slice=slice(None), show_y_lines=True,
            x_slice=slice(None), show_x_lines=True,
            show_colorbar=True, show_xticklabesl=False,
            show_axl=True, show_axb=True,
            **kwargs):
    """
    Contour plot for a given `TimeResolved` obj. This also plots the
    summations of the projections in x and y direction.

    The y and x projections show the mean Value throughout the given region.

    Parameters
    ----------
    x: array
        pp_delays
    y: array
        wavenumbers or pixel
    z : 2d array with the shape of (x.shape,y.shape)
        bleach
    N : Optional [int]
        Number of Contour lines
    fig: figure to draw on.
    x_slice / y_slice: slice or iterable.
        If slice, the slice is directly applied to the summation.
        If slice is a iterable it will be used as edges of a slice in
        plot coordinates
    show_y_lines / show_x_lines: Boolean default True
        if True and x_slice or y_slice is given, lines that show the slices are
        plotted.
    show_colorbar: boolean
        if true show a colorbar
    show_axl: show the y-projection (bleach) of the data
    **kwargs
        passed to contour plot see documentation of
        `matplotlib.pyplot.contourf`

    Returns
    -------
    matplotlib.fig
        The figure of the plot
    tuple of matpotlib.axes
        The three axes of the three subplots.
    """
    if not isinstance(x_slice, slice):
        x_slice = slice(*get_interval_index(x, *x_slice ))
    if not isinstance(y_slice, slice):
        y_slice = slice(*get_interval_index(y, *y_slice))

    # prepare figure and axes
    if not fig:
        fig = plt.figure()
    # I need a array structured ax return for that to work
    if show_axl and show_axb:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
        ax = plt.subplot(gs[0, 1])
        axl = plt.subplot(gs[0, 0], sharey=ax)
        axb = plt.subplot(gs[1, 1], sharex=ax)
    elif show_axb:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = plt.subplot(gs[0, 0])
        axb = plt.subplot(gs[1, 0], sharex=ax)
        axl = None # So we can allways return the same shape
    elif show_axb:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
        ax = plt.subplot(gs[0, 1])
        axb = None
        axl = plt.subplot(gs[0, 0], sharey=ax)
    else:
        ax = plt.gca()
        axl = None
        axb = None
        #gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        #ax = plt.subplot(gs[0, 0])
        #axl = None
        #axb = None

    # the actual plot
    CS = ax.contourf(x, y, z.T, N, **kwargs)
    if show_colorbar:
        plt.colorbar(CS, ax=ax)
    if x_slice != slice(None) and show_x_lines:
        ax.vlines(x[[x_slice.start, x_slice.stop]], y.min(), y.max(), linestyles="dashed")
    if y_slice != slice(None) and show_y_lines:
        ax.hlines(y[[y_slice.start, y_slice.stop]], x.min(), x.max(), linestyles="dashed")

    if show_axl:
        xl_data = z[x_slice].mean(0)
        axl.plot(xl_data, y)
        axl.set_xlim(xl_data.max(), xl_data.min())
        #axl.set_xticklabels(axl.get_xticks(), rotation=-45)

    if show_axb:
        axb.plot(x, z[:,y_slice].mean(1), "-o")
        axb.set_xlim(x.min(), x.max())
        #axb.locator_params(axis='y', nbins=4)

    if not show_xticklabesl:
        if show_axl:
            plt.setp(ax.get_yticklabels(), visible=False)
        if show_axb:
            plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    return fig, ax, axl, axb
