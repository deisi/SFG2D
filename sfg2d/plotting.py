from os import path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
import numpy as np

from .utils.filter import double_resample


def ioff(func):
    """Decorator to make plotting non interactive temporally ."""
    def make_ioff(*args, **kwargs):
        plt.ioff()
        func(*args, **kwargs)
        plt.ion()

    return make_ioff


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

    print("Saved figure to: {}".format(path.abspath(fpath)))


def plot_spec(xdata, ydata, ax=None, **kwargs):
    """
    Plot data with pixel axis of ydata as x-axis

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
                ax.plot(xdata, spec.T, **kwargs)


def plot_trace(xdata, ydata, ax=None, **kwargs):
    """
    data is the result of a subselection.

    This plot has delays on its x-axis
    """
    if not ax:
        ax = plt.gca()

    # Transpose because we want the delay axis to be the last axis
    # of the array.
    y = ydata.T
    for pixel in y:
        for spec in pixel:
            for frame in spec:
                ax.plot(xdata, frame.T, **kwargs)


def plot_contour(
        record,
        contour_kws={},
        plot_kws={},
        colorbar=True,
        xlabel='Time in fs',
        ylabel='Wavenumber in 1/cm',
):
    """Make Contour plot.

    Uses the SfgRecord.contour method to get data. contour_kws get passed
    to SfgReco.contour, plot_kws get passed to plt.contourf.

    If no levels for plot_kws are given, a 20 to 80 percentile with 15 levels is used.
    """
    plot_kws.setdefault('extend', 'both')
    x, y, z = record.contour(**contour_kws)

    if isinstance(plot_kws.get('levels'), type(None)):
        start, stop = [np.percentile(z.flatten(), perc) for perc in (1, 80)]
        levels = np.linspace(start, stop, 15)
        plot_kws['levels'] = levels

    plt.contourf(x, y, z, **plot_kws)
    if colorbar:
        plt.colorbar()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


def bleach_plot_slider(
        record,
        y_property='bleach_abs',
        x_property="wavenumber",
        fig=None,
        ax=None,
        l_kwgs={"loc": "lower left"},
        ylim=None,
        xlim=None,
        **kwargs
):
    """Bleachplot, with slidable pp_delay index and autoscale.

    record: SfgRecord to plot.
    y_property: bleach property to select data from.
    x_property: x data property
    fig: figure
    ax: axes
    l_kwgs: Keywordsfor the plot legend
    ylim: Optional tubple. Set fix ymin and ymax.
    **kwargs are passed to *SfgRecord.plot_bleach*
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
        record.plot_bleach(
            ax=ax,
            y_property=y_property,
            rois_delays=[slice(index, index+1)],
            x_property=x_property,
            label="{} fs".format(record.pp_delays[index]),
            **kwargs
        )

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
def bleach_plotzt_pdf(
        record,
        sfile,
        sfolder="./figures/",
        y_property='bleach_abs',
        x_property="wavenumber",
        xlim=None,
        ylim=None,
        num_base='bl{}',
        xlabel='Wavenumber in 1/cm',
        ylabel=None,
        l_kwgs={"loc": "lower left"},
        title_prefix=None,
        medfilt_pixel=5,
        plot_kwgs={},
        scale=1,
        delay_offset=0,
        **kwargs
):
    """Multipage pdf for the bleach plot.

    axes limits are always the same for all subplots.

    record: SfgRecord
    sfile: path to save the file to
    num_base: str that can be filled with format.
        used to name the figures.
    """

    # Makte ion and ioff use a decorator.
    subselect_kws = dict(**kwargs)
    subselect_kws.setdefault('frame_med', True)
    subselect_kws.setdefault('x_property', x_property)
    subselect_kws.setdefault('y_property', y_property)
    subselect_kws.setdefault('medfilt_pixel', medfilt_pixel)
    figs = []
    for index in range(record.number_of_pp_delays):
        fig, ax = plt.subplots(num=num_base.format(index))
        figs.append(fig)

        x, y = record.subselect(
            roi_delay=slice(index, index+1),
            **subselect_kws,
        )
        plot_spec(x, scale*y, ax=ax, **plot_kwgs)

        #record.plot_bleach(
        #    ax=ax,
        #    y_property=y_property,
        #    rois_delays=[slice(index, index+1)],
        #    x_property=x_property,
        #    label="{} fs".format(record.pp_delays[index]),
        #    **kwargs
        #)

        if not title_prefix:
            title_prefix = record.metadata('material')
        ax.set_title("{} @ {} fs".format(
            title_prefix, record.pp_delays[index]+delay_offset)
        )
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
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


def time_track(
        record,
        ax=None,
        xlim=None,
        ylim=None,
        title='Time Track of Scan',
        **kwargs
):
    """Plot a time track.

    For kwargs see *SfgRecod.plot_time_track*
    """
    if not ax:
        ax = plt.gca()

    record.plot_time_track(ax=ax, **kwargs)
    plt.vlines(
        record.number_of_pp_delays*np.arange(record.number_of_frames),
        ax.get_ylim()[0],
        ax.get_ylim()[1],
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)


def frame_track(
        record,
        ax=None,
        y_property='basesubed',
        frame_track_kwg={},
        **kwargs
):
    """Sane default for plotting a frame track.

    frame_track_kwg are passed to `SfgRecord.frame_track`"""
    if y_property:
        frame_track_kwg["y_property"] = y_property
    if not ax:
        ax = plt.gca()
    data = record.frame_track(**frame_track_kwg)
    ax.plot(data, "-o", **kwargs)
    ax.set_title("Mean Signal vs Frame Number")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Mean Counts")






# DEPRECATED
def spec_plot(
    record,
    ax=None,
    xlabel="Wavenumber 1/cm",
    ylabel="Counts",
    title="",
    x_property="wavenumber",
    y_property="basesubed",
    **kwargs
):
    """Plot Wrapper."""
    if not ax:
        ax = plt.gca()

    record.plot_spec(
        ax=ax,
        x_property=x_property,
        y_property=y_property,
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def trace_plot(
        record,
        ax=None,
        xlabel="Time in fs",
        ylabel="Mean Counts",
        title=None,
        y_property="traces_basesubed",
        plt_kwgs={},
):
    if not ax:
        ax = plt.gca()

    if isinstance(title, type(None)):
        title = "Trace {} Pumped @ {} 1/cm".format(
            record.metadata.get('material'),
            record.metadata.get("pump_freq")
        )

    record.plot_trace(ax=ax, y_property=y_property, **plt_kwgs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


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


def contour(
        record,
        pixel_med=3,
        N=30,
        fig=None,
        figsize=(9, 6),
        show_y_lines=True,
        show_x_lines=True,
        show_colorbar=True,
        show_xticklabesl=False,
        show_axl=True,
        show_axb=True,
        show_axr=True,
        rois_x_pixel_trace=None,
        **kwargs
):
    """
    Contour plot for a given `TimeResolved` obj. This also plots the
    summations of the projections in x and y direction.

    The y and x projections show the mean Value throughout the given region.

    Parameters
    ----------
    record: SfgRecord obj to get data from
    pixel_med: Optional
        Amount of pixels to run a median filter over. Must be an odd number.
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
    levels: Optional
    **kwargs
        passed to contour plot see documentation of
        `matplotlib.pyplot.contourf`

    Returns
    -------
    matplotlib.fig
        The figure of the plot
    ax: The main axis of the contour plot
    axl: The left axis. The plot of the bleach
    axb: The bottom axis. The traces
    axr: The right axis: The normalized static spectrum.
    """

    # Prepare the data.
    x = record.pp_delays
    y = record.wavenumber  # only use a pixel subset.
    z = np.median(
            record.bleach_rel[
                :,
                record.roi_frames,
                0,
                record.roi_x_pixel_spec
            ],
            1
        )
    if pixel_med:
        z = medfilt(z, (1, pixel_med))
    if N:
        z = double_resample(z, N, 1)
    xx = x[record.roi_delay]
    yy = y[record.roi_x_pixel_spec]
    zz = z[record.roi_delay]

    # prepare figure and axes
    ax, axr, axb, axl = None, None, None, None
    if not fig:
        fig = plt.figure(figsize=figsize)

    # I need a array structured ax return for that to work
    if show_axl and show_axb and show_axr:
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 3, 1], height_ratios=[3, 1])
        ax = plt.subplot(gs[0, 1])
        axl = plt.subplot(gs[0, 0], sharey=ax)
        axb = plt.subplot(gs[1, 1], sharex=ax)
        axr = plt.subplot(gs[0, 2], sharey=ax)
    elif show_axb:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = plt.subplot(gs[0, 0])
        axb = plt.subplot(gs[1, 0], sharex=ax)
        axr = None
        axl = None  # So we can always return the same shape
    elif show_axl:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
        ax = plt.subplot(gs[0, 1])
        axb = None
        axl = plt.subplot(gs[0, 0], sharey=ax)
    else:
        ax = plt.gca()
        axl = None
        axb = None

    # the actual plot
    CS = ax.contourf(
        xx,
        yy,
        zz.T,
        N,
        extend="both",
        **kwargs
    )
    if show_colorbar:
        plt.colorbar(CS, ax=ax)

    if show_axl:
        for index in range(len(record.rois_delays_pump_probe)):
            roi_delay = record.rois_delays_pump_probe[index]
            y_axl = yy
            x_axl = record.subselect(
                y_property='bleach_rel',
                roi_delay=roi_delay,
                frame_med=True,
                delay_mean=True,
            )[1].squeeze()
            axl.plot(x_axl, y_axl)
            color = axl.get_lines()[-1].get_color()
            x_axl = x[roi_delay]
            if roi_delay != slice(None) and show_x_lines:
                ax.vlines(
                    [x_axl.min(), x_axl.max()],
                    yy.min(),
                    yy.max(),
                    linestyles="dashed",
                    color=color
                   )
        #if axl.get_xlim()[0] < 0:
        #     axl.set_xlim(left=0)
        #if axl.get_xlim()[1] < 1.2:
        #     axl.set_xlim(right=1.2)

    if show_axr:
        y_axr = yy
        x_axr = record.subselect(
            y_property="unpumped_norm",
            frame_med=True,
            delay_mean=True
        )[1].squeeze()
        axr.plot(x_axr, y_axr)
        color = axr.get_lines()[-1].get_color()

    if show_axb:
        trace_plot(record, ax=axb, y_property="traces_bleach_rel")
        for index in range(len(record.rois_x_pixel_trace)):
            roi_x_pixel_trace = record.rois_x_pixel_trace[index]
            color = axb.get_lines()[index].get_color()
            y_axb = y[roi_x_pixel_trace]
            if record.rois_x_pixel_trace != slice(None) and show_y_lines:
                for ax_elm in (ax, axl, axr):
                    if not ax_elm:
                        continue
                    # Why is this needed. Doesn't make sense.
                    if ax_elm == ax:
                        xmin, xmax = xx[0], xx[-1]
                    else:
                        xmin, xmax = ax_elm.get_xlim()
                    ax_elm.hlines(
                        [y_axb.min(), y_axb.max()],
                        xmin, xmax,
                        linestyles="dashed", color=color
                       )
                    ax_elm.set_xlim(xmin, xmax)

    if not show_xticklabesl:
        if show_axl:
            plt.setp(ax.get_yticklabels(), visible=False)
        if show_axb:
            plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    return fig, ax, axl, axb, axr
