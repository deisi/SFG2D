"""Physical Model that can be applied to the data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from .utils.static import gaus_func

class CurveFitter():
    """Base Class to add curve fit capabilities to model classes."""
    def __init__(self, xdata, ydata, p0=None):
        self.xdata = xdata
        self.ydata = ydata
        self.p0 = p0  # Initial fit values
        self.p = None  # The fit result parameters
        self.cov = None  # The covariance of the fit

    def fit_func(self, x, *args, **kwargs):
        # Overwrite this in the children class
        raise NotImplementedError

    def curve_fit(self, **kwargs):
        """Make a least square fit"""
        self.p1, self.cov = curve_fit(
            self.fit_func,
            self.xdata,
            self.ydata,
            self.p0,
            **kwargs
        )
        return self.p1, self.cov

    def fit_res(self, x):
        """Fit function wit fit result parameters"""
        return self.fit_func(x, *self.p1)

    @property
    def yfit(self):
        return self.fit_res(self.xdata)

    def plot(self, fig=None, ax=None,
             show_fit_points=False, show_fit_line=True, number_of_samples=100):
        if not fig:
            fig = plt.gcf()
        if not ax:
            ax = plt.gca()

        ax.plot(self.xdata, self.ydata, "-o", label='data')
        if show_fit_points:
            ax.plot(self.xdata, self.yfit, "-o", label='fit')
        if show_fit_line:
            x_sample = np.linspace(self.xdata.min(), self.xdata.max(), 100)
            ax.plot(x_sample, self.fit_res(x_sample), label="fit")


class FourLevelMolKin(CurveFitter):
    def __init__(self, xdata, ydata, p0=None, gSigma=300):
        """Class for the four level Molecular Dynamics Model.

        Parameters
        ----------
        gSigma: width of the excitation pulse in fs.
        """
        super().__init__(xdata, ydata, p0)
        self.gSigma = gSigma

    # Gausian Excitation Function.
    def ext_func(self, t, mu):
        """Gausian excitation function with fixed with.

        Parameters:
        t: time in fs
        mu: start of the pump in fs

        Returns:
        array of values on a gaussian function.
        """
        return gaus_func(t, 1, mu, self.gSigma)

    # The Physical Water model
    def dgl(self, N, t, ext_func, s, t1, t2):
        """Dgl of the 4 Level DGL system.

        Parameters:
        -----------
        N : deg 4 array
            Population of the 4 levels respectively
        t : float
            time
        ext_func : exictation function in time.
            Time profile of the pump laser.
            Function of t. Usaully a gaussian function.
        s : scaling factor of the pump laser.
        t1 : Time constant of first level
        t2 : Time constant of second level.

        Returns
        -------
        Derivatices of the system. As 4 dim array.
        1 dimension per level."""
        A = np.array([
            [-s * ext_func(t), s * ext_func(t), 0, 0],
            [s * ext_func(t), -s * ext_func(t) - 1/t1, 0, 0],
            [0, 1 / t1, -1 / t2, 0],
            [0, 0, 1 / t2, 0],
        ], dtype=np.float64)
        dNdt = A.dot(N)
        return dNdt

    def population(self, t, ext_func, s, t1, t2, rtol=1.09012e-9, **kwargs):
        """Numerical solution to the 4 Level DGL-Water system.
        Parameters
        ----------
        t: array of time values
        ext_func: Function of excitation.
        s: scalar factor for the pump
        t1: Live time of the first exited state
        t2: livetime of the intermediate state.
        rtol: precisioin of the numerical integrator.
            default if scipy is not enough. The lower
            this number the more exact is the solution
            but the slower the function.

        Returns
        -------
        (len(tt), 4) shaped array with the 4 entires beeing the population
        of the N0 t0  N3 levels of the system
        """
        ret = odeint(
            self.dgl,  # the DGL of the 3 level water system
            [1, 0, 0, 0],  # Starting conditions of the DGL
            t,
            args=(ext_func, s, t1, t2),
            rtol=rtol,
            **kwargs,
        )
        return ret

    # The Function for the Trace that drops out of the Model
    def fit_func(self, t, s, t1, t2, c, mu):
        """
        Function we use to fit.

        parameters
        ----------
        t: time
        s: Gaussian Amplitude
        t1: Livetime of first state
        t2: livetime of second state
        c: Coeficient of 3rd state
        scale: Scaling factor at the very end

        Returns:
        The bleach of the water model
        and the Matrix with the populations"""
        N = self.population(
            t,
            lambda t: gaus_func(t, 1, mu, self.gSigma),
            s,
            t1,
            t2
        ).T
        # Transposing makes N[0] the population of state 0 and so forth.
        return ((N[0] + N[2] - N[1] + c * N[3])**2) / (N[0]**2)
