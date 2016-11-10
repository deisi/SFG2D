import matplotlib.pyplot as plt

from . import ts
from . import scan


def plot(*args, **kwargs):
    """Wrapper for default matplotlib.plot that adds xlabel as wavenumbers"""
    plt.plot(*args, **kwargs)
    plt.xlabel('Wavenumbers in cm$^{-1}$')


def plot_time(time, data, **kwargs):
    """ Wrapper function to plot formatted time on the x-axis
    and data on the y axis. If time is datetime obj, the time is
    plottet in HH:MM, if time is timedelta it is plotted as minuits

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
    Plot errorbar as shadow around the data

    Parameters
    ----------
    x : array
        x-data
    y : array
        y-data
    dy : array
        uncertaincy of the y data
    ax : Optional [matplotlib.axes obj]
        the aces to plot on to.
    **kwargs :
        kwargs are passed to `matplotlib.plot`
    """
    if not ax:
        ax = plt.gca()
    ax.plot(x, y, color=color, **kwargs)
    ax.fill_between(x, y-dy, y+dy, color=color, alpha=0.5)






