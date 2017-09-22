from os import path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
import numpy as np

from .utils.filter import double_resample
from .utils.consts import x_property2_label


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


def plot_spec(xdata, ydata, ax=None, xlabel='Wavenumber in 1/cm', ylabel='SFG Intensity in a.u.', **kwargs):
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

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_trace(xdata, ydata, ax=None, yerr=None, xlabel='Time in fs', ylabel='Bleach in a.u.', **kwargs):
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


def plot_contour(
        x, y, z,
        colorbar=True,
        xlabel='Time in fs',
        ylabel='Wavenumber in 1/cm',
        levels=np.linspace(0.8, 1.1, 15),
        **kwgs
):
    """Make Contour plot.

    If no levels kwgs is given, a 5 t0 80 percentile with 15 levels is used.
    colorbar: boolean for colorbar
    xlabel: string for xlabel
    ylabel: string for ylabel

    kwgs are passed to *plt.contourf*
    """
    kwgs.setdefault('extend', 'both')

    if isinstance(kwgs.get('levels'), type(None)):
        kwgs['levels'] = levels

    plt.contourf(x, y, z, **kwgs)
    if colorbar:
        plt.colorbar()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


def plot_trace_fit(
        xdata,
        ydata,
        yerr=None,
        fit_func=None,
        box_str=None,
        box_coords=None,
        x_fit_range=None,
        y_fit_range=None,
        xsample=400,
        ax=None,
        xlabel='Time in fs',
        ylabel='Relative Bleach',
        data_kws={},
        fit_kws={},
        text_kws={},
):
    """
    xdata: data of the x axis
    ydata: data of the y axis
    yerr: yerr of the data
    fit_func: function of xdata that is the fit result
    x_fit_range: xmin and xmax of the effective fit region
    y_fit_range: ymin and ymax of the effective fit region
    xsample: number of samples to plot the fit func with
    ax: axes obj to draw on
    xlabel: xlabel of the plot
    ylabel: ylabel of the plot
    box_str: string of fit parameters to plot
    box_coords: tuple coordinates of the parameter box
    data_kws: keywords passed to the data plot.
    fit_kws: keywords passed to the plot of the fit.
    text_kws: keywords passed to the text box
    """
    if not ax:
        ax = plt.gca()
    else:
        plt.sca(ax)

    data_kws.setdefault('label', 'Data')
    if not isinstance(yerr, type(None)):
        data_kws.setdefault('fmt', 'o')
        plt.errorbar(xdata, ydata, yerr, **data_kws)
    else:
        ax.plot(xdata, ydata, **data_kws)

    if fit_func:
        xsample = np.linspace(xdata[0], xdata[-1], xsample)
        fit_kws.setdefault('label', 'Fit')
        ax.plot(xsample, fit_func(xsample), **fit_kws)

    if not isinstance(x_fit_range, type(None)) and not isinstance(y_fit_range, type(None)):
        ax.scatter(x_fit_range, y_fit_range, color='r', zorder=3)

    if box_str:
        text_kws.setdefault('fontdict', {})
        text_kws['fontdict'].setdefault('family', 'monospace')
        if isinstance(box_coords, type(None)):
            box_coords = xdata.mean(), ydata.mean()
        ax.text(*box_coords, box_str, **text_kws)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


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

    select_y_kws = dict(**kwargs)
    select_y_kws.setdefault('frame_med', True)
    select_y_kws.setdefault('prop', y_property)
    select_y_kws.setdefault('medfilt_pixel', medfilt_pixel)
    figs = []
    for index in range(record.number_of_pp_delays):
        fig, ax = plt.subplots(num=num_base.format(index))
        figs.append(fig)

        y = record.select(
            roi_delay=slice(index, index+1),
            **select_y_kws,
        )
        x = record.select(prop=x_property, roi_pixel=kwargs.get('roi_pixel'))
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


# Plots on Records.
def plot_record_static(record, save=True, scale=1000, select_kw={}):
    """Figure of Static data from a record.

    High level function.

    record: Record to get data from
    save: Boolean, Save figure
    scale: Scale y axis.
    select_kw: dict passed to select method

    Returns
      fig and ax.
    """
    fig, ax = plt.subplots(num='{}_static'.format(record.name))
    fig.clf()
    select_kw.setdefault('delay_mean', True)
    select_kw.setdefault('frame_med', True)
    select_kw.setdefault('prop', 'unpumped')
    data = record.select(**select_kw)
    plot_spec(record.select('wavenumber'), scale*data)
    plt.title("{}".format(record.lname))
    fname = 'figures/{}_static.pdf'.format(record.name)
    print(fname)
    if save:
        plt.savefig(fname)
        print("saved")
    return fig, ax


def plot_record_contour(
        record,
        levels=(90, 105, 15),
        save=True,
        xlim=(-1000, 5000),
        fname_form='./figures/{}_contour_bleach_rel_pump{}.pdf',
):
    fig, ax = plt.subplots(
        figsize=1*np.array([10, 6]), num='{}_contour'.format(record.name)
    )
    fig.clf()
    x, y, z = record.contour(
        z_property='bleach',
        prop_kwgs={'opt': 'rel', 'prop': 'basesubed'},
        resample_freqs=30,
        medfilt_pixel=7
    )
    plot_contour(x, y, 100*z, levels=np.linspace(*levels))
    plt.title("{} Pump @ {} 1/cm".format(record.lname, record.pump_freq))
    plt.tight_layout()
    plt.xlim(*xlim)
    if save:
        fname = fname_form.format(
            record.name, record.pump_freq
        )
        print('Saving to {}'.format(fname))
        plt.savefig(fname)
    return fig, ax


def plot_model_trace(
        name,
        model,
        record,
        xlim=(-1200, 5100),
        ylim=(0.90, 1.02),
        save=True,
        fname_format='figures/{}_trace_bleach_rel_pump{}_{}_fit.pdf',
):
    fig, ax = plt.subplots(
        num='{}'.format(name)
    )
    fig.clf()
    plt.title("{} {} 1/cm".format(record.lname, name))
    plot_trace_fit(
        model._xdata,
        model._ydata,
        model._sigma,
        model.fit_res,
        model.box_str,
        model.box_coords,
        model.x_edges,
        model.y_edges,
    )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    fname = fname_format.format(
        record.name, record.pump_freq, name)
    print('Filename: ', fname)
    if save:
        print("Saved")
        plt.savefig(fname)
    return fig, ax


# DEPRECATED
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

