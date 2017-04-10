import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from . import ts
from . import scan
from ..utils import get_interval_index

# decorator function to save loopable plot in multipage pdf
def multipage_pdf(plot_func):
    """
    Function can be used as a decorator, to loop over a given range
    and call a plot function with an index parameter in each loop
    incerement. E.g.

    ```
    @sfg2d.multipage_pdf
    def my_plot(index):
        record = SfgRecord()
        record.plot_bleach(attribute="bleach", x_axis="wavenumber", pp_delays=[index]);
        title(r"Sample @ %i fs"%record.pp_delays[index])
        xlim(2100, 2800)
        ylim(-0.003, 0.002)

    my_plot("delme", range(2))
    ```


    plot_func: function
        ploting function with index value as parameter.
        each time the index is increased and plot_func is called
        with the new index again.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    def make_multipage_pdf(name, inds):
        """
        fname: string
            filename to save the multipage pdf in.
        indes: iterable
            iterable to loop over
        """
        if name[-4:] != '.pdf':
            name += '.pdf'
        with PdfPages(name) as pdf:
            for index in inds:
                fig, ax = plt.subplots()
                plot_func(index)
                pdf.savefig()
                plt.close()

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

def errorshadow(x, y, dy, ax=None, color="b", **kwargs):
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
    ax.plot(x, y, color=color, **kwargs)
    ax.fill_between(x, y-dy, y+dy, color=color, alpha=0.5)

def contour(x, y, z, N=30, fig=None,
            y_slice=slice(None), show_y_lines=True,
            x_slice=slice(None), show_x_lines=True,
            show_colorbar=True, show_xticklabesl=False,
            **kwargs):
    """
    Contour plot for a given `TimeResolved` obj. This also plots the
    summations of the projections in x and y direction.

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
        of slice is a iterable it will be used as edges of a slice in
        plot coordinates
    show_y_lines / show_x_lines: Boolean default True
        if True and x_slice or y_slice is given, lines that show the slices are
        plotted.
    show_colorbar: boolean
        if true show a colorbar
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
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
    ax = plt.subplot(gs[0, 1])
    axl = plt.subplot(gs[0, 0], sharey=ax)
    axb = plt.subplot(gs[1, 1], sharex=ax)

    # the actual plot
    CS = ax.contourf(x, y, z.T, N, **kwargs)
    if show_colorbar:
        plt.colorbar(CS, ax=ax)
    if x_slice != slice(None) and show_x_lines:
        print("x still ture")
        print(x_slice)
        ax.vlines(x[[x_slice.start, x_slice.stop]], y.min(), y.max(), linestyles="dashed")
    if y_slice != slice(None) and show_y_lines:
        print("y still ture")
        print(y_slice)
        ax.hlines(y[[y_slice.start, y_slice.stop]], x.min(), x.max(), linestyles="dashed")

    xl_data = z[x_slice].sum(0)
    axl.plot(xl_data, y)
    axl.set_xlim(xl_data.max(), xl_data.min())
    #axl.set_xticklabels(axl.get_xticks(), rotation=-45)

    axb.plot(x, z[:,y_slice].sum(1), "-o")
    axb.set_xlim(x.min(), x.max())
    #axb.locator_params(axis='y', nbins=4)

    if not show_xticklabesl:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()
    return fig, ax, axl, axb


def img(x, y, z, *args, fig=None, ax=None, extent=None, aspect=20, method='cubic',
        **kwargs):
    """
    BROKEN
    Parameters
    ----------
    x :

    y :

    z : 
    
    *args
        Passed to `matplotlib.pyplot.imshow`
    extent : Optional [tuple with (xmin, xmax, ymin, ymax)]
        extent of imshow
    aspect : Optional [int]
        aspect of imshow
    method : str
        interpolation method used by griddata to interpolate points of the
        image inbetween measurment points valid options are:
        'nearest', 'linear' and 'cubic'.
        See SFG.example_analysis.AnalyseATimeResolvedSpectrum for examples.
    **kwargs      are passed to matplotlib.pyplot.imshow

    Returns
    -------
    matplotlib.fig
        The figure of the plot
    tuple of matpotlib.axes
        The three axes of the three subplots."""
    from itertools import product
    from scipy.interpolate import griddata

    if not fig:
        fig = plt.gcf()
    if not ax:
        ax = plt.gca()
    # Coordinates into the right form
    if not extent:
        ext = (x.min(), x.max(), y.min(), y.max())  # correct ticklabels
    points = np.array(list(product(x, y)))
    values = z.flatten()
    xi = np.transpose(np.mgrid[x.min():x.max():200j, y.min():y.max():200j])

    # interpolates between measured points
    grid = griddata(points, values, xi, method=method)

    ax.imshow(grid, *args, origin="lower", extent=ext, aspect=aspect, **kwargs)
