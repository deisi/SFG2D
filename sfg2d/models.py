"""Physical Model that can be applied to the data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sfg2d.utils.static import gaus_func

class CurveFitter():
    """Base Class to add curve fit capabilities to model classes."""
    def __init__(self, xdata, ydata, p0=None, bounds=None):
        self.xdata = xdata
        self.ydata = ydata
        self.p0 = p0  # Initial fit values
        self.p = None  # The fit result parameters
        self.cov = None  # The covariance of the fit
        self.jac = None  # The jacobean Matrix of the DGL
        self.bounds = bounds  # Boundary conditions for the fit

    def fit_func(self, x, *args, **kwargs):
        # Overwrite this in the children class
        raise NotImplementedError

    def curve_fit(self, **kwargs):
        """Make a least square fit"""
        if not kwargs.get("bounds"):
            kwargs["bounds"] = self.bounds
        self.p, self.cov = curve_fit(
            self.fit_func,
            self.xdata,
            self.ydata,
            self.p0,
            **kwargs
        )
        return self.p, self.cov

    def fit_res(self, x):
        """Fit function wit fit result parameters"""
        return self.fit_func(x, *self.p)

    @property
    def yfit(self):
        """y-data of the fit result."""
        return self.fit_res(self.xdata)

    @property
    def yfit_start(self):
        """y-data of the starting values."""
        return self.fit_func(self.xdata, *self.p0)

    @property
    def pnames(self):
        """Returns names of the fit parameters."""
        from iminuit import describe
        return describe(self.fit_func)[1:]

    @property
    def pdict(self):
        """dict with the fit result"""
        ret = dict(zip(self.pnames, self.p))
        for i in range(len(self.pnames)):
            pname = self.pnames[i]
            perror = pname + '_error'
            ret[perror] = self.cov[i, i]**2
        return ret

    def plot(self,
             fig=None, ax=None,
             show_start_curve=False,
             show_fit_points=False,
             show_fit_line=True,
             number_of_samples=100,
             show_box=True,
    ):
        """Convenience function to show the results.

        fig: figure object to draw on.
        ax: axis object to draw on.
        show_start_curve: boolean
            shows the starting curve.
        show_fit_points: boolean
            shows the result of the fit at the xdata points.
        show_fit_line: boolean
            makea smooth line from the fit result.
        number_of_points: number of data points for the smooth lines.
        show_box: boolean
            Show nummeric fit result as a text box on the plot."""
        if not fig:
            fig = plt.gcf()
        if not ax:
            ax = plt.gca()

        if show_box:
            text = ''
            for i in range(len(self.pnames)):
                pname = self.pnames[i]
                pvalue = self.pdict[pname]
                perror = self.pdict[pname + '_error']
                text += '{}: {:.3G} $\pm$ {:.1G}\n'.format(pname, pvalue, perror)
            ax.text(0.01, 0.3, text,
                    #horizontalalignment='center',
                    #verticalalignment='center',
                    transform=ax.transAxes)

        ax.plot(self.xdata, self.ydata, "-o", label='data')
        if show_start_curve:
            x_sample = np.linspace(self.xdata.min(), self.xdata.max(), number_of_samples)
            ax.plot(x_sample, self.fit_func(x_sample, *self.p0))
        if show_fit_points:
            ax.plot(self.xdata, self.yfit, "-o", label='fit')
        if show_fit_line:
            x_sample = np.linspace(self.xdata.min(), self.xdata.max(), number_of_samples)
            ax.plot(x_sample, self.fit_res(x_sample), label="fit")



class FourLevelMolKin(CurveFitter):
    def __init__(self, xdata, ydata, p0=None, gSigma=300, rtol=1.09012e-9,
                 full_output=True, metadata={}):
        """Class for the four level Molecular Dynamics Model.

        Parameters
        ----------
        xdata: array
            The xdata of the model
        ydata: array
            The ydata of the model
        p0: Starting values for the model
        gSigma: width of the excitation pulse  in  the same
            units as xdata.
        rtol: precisioin of the numerical integrator.
            default if scipy is not enough. The lower
            this number the more exact is the solution
            but the slower the function.
        full_output: If true create infodict for odeint
            result is saved under self.infodict
        metadata:
            dictionary for metadata.
        """
        super().__init__(xdata, ydata, p0)
        self.gSigma = gSigma  # width of the excitation
        self.rtol = rtol  # Precition of the numerical integrator.
        self.full_output = full_output
        self.infodict = None  # Infodict return of the Odeint.
        self.metadata = metadata # Dictionary to add metadata to


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
        Derivatives of the system. As 4 dim array.
        """

        # This is the DGL written as a Matrix multiplication.
        # dNdt = A x N
        # With A beeing the constructing matrix of the DGL
        # and N beeing a 4-level vector with (N0, N1, N2, N3)
        # beeing the population of the states at the time t.
        # dNdt is the state wise derivative of N
        # See https://en.wikipedia.org/wiki/Matrix_differential_equation
        # for further insights.

        A = np.array([
            [-s * ext_func(t), s * ext_func(t), 0, 0],
            [s * ext_func(t), -s * ext_func(t) - 1/t1, 0, 0],
            [0, 1 / t1, -1 / t2, 0],
            [0, 0, 1 / t2, 0],
        ], dtype=np.float64)
        dNdt = A.dot(N)
        return dNdt

    def jac(self, N, t, ext_func, s, t1, t2):
        """Jacobean matrix of the DGL."""
        # In this case the Jacobean Matrix is euqal the
        # Consturcting matrix of the DGL.
        # So it doesn't help much. It just speeds up the thing
        # a little.
        A = np.array([
            [-s * ext_func(t), s * ext_func(t), 0, 0],
            [s * ext_func(t), -s * ext_func(t) - 1/t1, 0, 0],
            [0, 1 / t1, -1 / t2, 0],
            [0, 0, 1 / t2, 0],
        ], dtype=np.float64)
        return A

    def population(self, t, ext_func, s, t1, t2, **kwargs):
        """Numerical solution to the 4 Level DGL-Water system.

        Parameters
        ----------
        t: array of time values
        ext_func: Function of excitation.
        s: scalar factor for the pump
        t1: Live time of the first exited state
        t2: livetime of the intermediate state.

        Returns
        -------
        (len(tt), 4) shaped array with the 4 entires beeing the population
        of the N0 t0  N3 levels of the system
        """

        ret = odeint(
            func=self.dgl,  # the DGL of the 3 level water system
            y0=[1, 0, 0, 0],  # Starting conditions of the DGL
            t=t,
            args=(ext_func, s, t1, t2),
            Dfun=self.jac,  # The Jacobean of the DGL. Its optional.
            rtol=self.rtol,  # The precisioin parameter for the nummerical DGL solver.
            full_output=self.full_output,
            **kwargs,
        )
        if self.full_output:
            ret, self.infodict = ret

        return ret

    # The Function of the Trace that drops out of the Model
    def fit_func(self, t, s, t1, t2, c, mu):
        """
        Function we use to fit.

        parameters
        ----------
        t: time
        s: Gaussian Amplitude
        t1: Livetime of first state
        t2: livetime of second(intermediate) state
        c: Coefficient of third(Heat) state
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
        return ((N[0] + N[2] + c * N[3] - N[1])**2) / (N[0]**2)
