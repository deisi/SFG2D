import matplotlib.pyplot as plt
import numpy as np


def sum_of_spectra(ts, spec, ax=None):
    """plot the sum of spectra per time delay

    Parameters
    ----------
    tds : sfg2d.core.scan.TimeScan
        the time scan to plot the sum of spectra from
    spec : str
        identifier of the spectra to look at. I.E. the name of the column
        e.g. 'spec_0'
    ax : axis
        axis obj to plot on. If None current is created or used.
    """

    if isinstance(ax, type(None)):
        ax = plt.gca()
    g = ts._df.sum(level='pp_delay')[spec]
    l_ppdelays = len(ts.pp_delays)
    exp = ts.metadata['exposure_time'][0]
    exp_unit = ts.metadata['exposure_time'][1]
    try:
        for i in range(g.shape[1]):
            x = np.arange(l_ppdelays)*exp
            x = x + l_ppdelays*exp*i
            ax.plot(x, g.iloc[:, i], label='run %i' % i)
    except IndexError:
        x = np.arange(l_ppdelays)*exp
        ax.plot(x, g, label='run %i' % 0)

    ax.set_xlabel(r'$\Delta$t/%s' % exp_unit)
    ax.set_title('Sum of Spectra for %s' % spec)


def bleach_spec(ts, pp_delay, w_roi, ax=None):
    """plot the bleach of a TimeScan at a pp_delay in w_roi.

    Parameters
    ----------
    ts : TimeScan
    pp_delay : int
        pump_probe_delay
    w_roi : slice
        region of interest in wavenumbers"""
    if isinstance(ax, type(None)):
        ax = plt.gca()
    ts.bleach[pp_delay][w_roi].plot(ax=ax)
    ax.set_title("%i fs" % pp_delay)
    ax.grid()


def bleach_sum(ts, roi=slice(None, None)):
    """plot the bleach by looking at the summation of the roi."""
    raise NotImplementedError
