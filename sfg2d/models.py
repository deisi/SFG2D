"""Physical Model that can be applied to the data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import norm

class Fitter():
    def __init__(
            self,
            xdata=None,
            ydata=None,
            p0=None,
            sigma=None,
            box_coords=None,
            roi=None,
    ):
        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma  # 1/y-errors.
        self.p0 = p0  # Initial fit values
        self.cov = None  # The covariance of the fit
        # Coordinates of the fit result box in fig coordinates
        self.box_coords = box_coords
        self._box_str_format = '{:2}: {:8.3g} $\pm$ {:6.1g}\n'
        if not roi:
            roi = slice(None)
        self.roi = roi
        self._pnames = None

    @property
    def xdata(self):
        return self._xdata[self.roi]

    @property
    def ydata(self):
        return self._ydata[self.roi]

    @property
    def sigma(self):
        return self._sigma[self.roi]

    @property
    def yerr(self):
        return self.sigma

    def fit_res(self, x):
        """Fit function wit fit result parameters"""
        return self.fit_func(x, *self.p)

    @property
    def x_edges(self):
        """Edges of the x data of the fit."""
        return self.xdata[0], self.xdata[-1]

    @property
    def y_edges(self):
        """Edges of the y data of the fit."""
        return self.ydata[0], self.ydata[-1]


class CurveFitter(Fitter):
    """Base Class to add curve fit capabilities to model classes."""
    def __init__(self, *args, bounds=(-np.inf, np.inf), metadata={},
                 fit_slice=(None, None, None), fname=None, **kwargs):
        Fitter.__init__(self, *args, **kwargs)
        self.jac = None  # The jacobean Matrix of the DGL
        self.bounds = bounds  # Boundary conditions for the fit
        self.metadata = metadata  # Some additional metadata
        self._xsample_num = 400
        # Effective slice of the fit to take into account.
        self.fit_slice = slice(*fit_slice)
        if fname:
            self.load(fname)

    def curve_fit(self, **kwargs):
        """Make a least square fit"""
        if not kwargs.get("bounds"):
            kwargs["bounds"] = self.bounds
        if not kwargs.get('sigam'):
            kwargs['sigma'] = self.sigma
        self.p, self.cov = curve_fit(
            self.fit_func,
            self.xdata[self.fit_slice],
            self.ydata[self.fit_slice],
            self.p0,
            **kwargs
        )
        return self.p, self.cov

    @property
    def perror(self):
        '''Error estimation of the fit parameters.'''
        return np.sqrt(self.cov.diagonal())

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
        if isinstance(self._pnames, type(None)):
            from iminuit import describe
            return describe(self.fit_func)[1:]
        else:
            return self._pnames

    @property
    def pdict(self):
        """dict with the fit result"""
        ret = dict(zip(self.pnames, self.p))
        for i in range(len(self.pnames)):
            pname = self.pnames[i]
            perror = pname + '_error'
            ret[perror] = np.sqrt(self.cov[i, i])
        return ret

    @property
    def X2(self):
        """The unreduced Chi2 of the Fit.

        This is the squaresum of data and fit points."""
        return np.square(self.yfit - self.ydata).sum()

    @property
    def X2_start(self):
        """The unreduced Chi2 of the starting parameters."""
        return np.square(self.yfit_start - self.ydata).sum()

    @property
    def xsample(self):
        return np.linspace(self.xdata[0], self.xdata[-1], self._xsample_num)

    @property
    def box_str(self):
        """String to place on the plot. Showing Fit Statistics."""
        text = ''
        for name, value, error in zip(self.pnames, self.p, self.perror):
            text += self._box_str_format.format(
                name, value, error
            )
        return text

    def plot(self,
             fig=None, ax=None,
             show_data=True,
             show_start_curve=False,
             show_fit_points=False,
             show_fit_line=False,
             number_of_samples=100,
             show_box=False,
             box_coords=None,
             show_fit_range=False,
             data_plot_kw={'linestyle': '-',
                           'label': 'data',
                           'marker': 'o'},
             start_plot_kw={},
             fitl_plot_kw={},
             fitp_plot_kw={},
             vlines_param_dict={},
             **kwgs

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
            Show nummeric fit result as a text box on the plot.
        show_fit_range: boolean
            show what data was used for the fit.
        box_coords: Optinal update the box_coords.
            box_coords are a tuple of (x,y) coords in
            axis coordinates, thus from 0 to 1
        kwgs get passed to all plots.
        """
        if not fig:
            fig = plt.gcf()
        if not ax:
            ax = plt.gca()

        x_sample = np.linspace(
            self.xdata.min(),
            self.xdata.max(),
            number_of_samples
        )

        if show_box:
            if not isinstance(box_coords, type(None)):
                self.box_coords = box_coords

            text = ''
            for i in range(len(self.pnames)):
                pname = self.pnames[i]
                pvalue = self.pdict[pname]
                perror = self.pdict[pname + '_error']
                text += '{:3}: {:.2G} $\pm$ {:.1G}\n'.format(
                    pname, pvalue, perror
                )
            ax.text(*self.box_coords, text,
                    transform=ax.transAxes)

        if show_data:
            if isinstance(self.sigma, type(None)):
                ax.plot(self.xdata, self.ydata, **data_plot_kw, **kwgs)
            else:
                plt.errorbar(self.xdata, self.ydata, yerr=self.sigma, axes=ax, **data_plot_kw)
        if show_start_curve:
            ax.plot(x_sample, self.fit_func(x_sample, *self.p0),
                    label="start", **start_plot_kw, **kwgs)
        if show_fit_points:
            ax.plot(self.xdata, self.yfit,
                        "-o", label='fit', **fitp_plot_kw, **kwgs)
        if show_fit_line:
            ax.plot(x_sample, self.fit_res(x_sample),
                    label="fit", **fitl_plot_kw, **kwgs)
        if show_fit_range:
            ax.vlines(
                [self.xdata[self.fit_slice.start],
                 self.xdata[self.fit_slice.stop]],
                self.ydata.min(), self.ydata.max(),
                vlines_param_dict
            )

    def save(self, fname, parameter_dict={}):
        """Save CurveFitter results.

        fname: File path.
        parameter_dict: Additional parameters to save."""
        np.savez_compressed(
            fname,
            xdata=self.xdata,
            ydata=self.ydata,
            p0=self.p0,
            p=self.p,
            cov=self.cov,
            bounds=self.bounds,
            metadata=self.metadata,
            box_coords=self.box_coords,
            fit_slice=self.fit_slice,
            **parameter_dict,
        )

    def load(self, fname):
        """Load a saved CurvedFitter obj."""
        inp = np.load(fname)
        for key, value in inp.items():
            if key == 'metadata':
                setattr(self, key, value[()])
                continue
            setattr(self, key, value)


class Minuitter(Fitter):
    """Base Class to use Minuit as Fitting backend."""
    def __init__(
            self,
            xdata=None,
            ydata=None,
            p0=None,
            sigma=None,
            fitarg={},
            **kwargs
    ):
        from iminuit import Minuit
        #from probfit import Chi2Regression
        super().__init__(xdata, ydata, p0, sigma, **kwargs)
        # TODO why doenst this work?
        #self.chi2 = Chi2Regression(
        #    self.fit_func,
        #    self.xdata,
        #    self.ydata,
        #    error=sigma,
        #   )
        self.minuit = Minuit(self.chi2, **fitarg, pedantic=False)

    @property
    def p0(self):
        return self.minuit.args

    @p0.setter
    def p0(self, value):
        self._p0 = value

    @property
    def p(self):
        return self.minuit.args

    @property
    def box_str(self):
        """String to place on the plot. Showing Fit Statistics."""
        text = ''
        for name, value in zip(self.minuit.parameters, self.minuit.args):
            text += self._box_str_format.format(
                name, value, self.minuit.errors[name]
            )
        return text


class FourLevelMolkinBase():
    def __init__(
            self,
            gSigma=150,
            N0=[1, 0, 0, 0],
            rtol=1.09012e-9,
            atol=1.49012e-9,
            full_output=True,
    ):
        """Baseclass of the 4 Level Molekular Dynamics Model."""
        self.gSigma = gSigma  # width of the excitation
        self.rtol = rtol  # Precition of the numerical integrator.
        self.atol = atol
        # Starting conditions of the Populations, not to be confuesed with starting conditions of the plot
        self.N0 = N0
        self.full_output = full_output
        self.infodict = None  # Infodict return of the Odeint.

    def ext_gaus(self, t, mu, sigma):
        """Gausian excitation function.

        Due to historic reasons its not a strict gausian, but something
        very cloe to it. The Igor Code is:
        1/sqrt(pi)/coeff1*exp(-(coeff0-x)^2/coeff1^2) """

        return 1 / np.sqrt(np.pi) / sigma * np.exp(-((mu-t)/sigma)**2)

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
        (len(t), 4) shaped array with the 4 entires beeing the population
        of the N0 t0  N3 levels of the system
        """

        ret = odeint(
            func=self.dgl,  # the DGL of the 3 level water system
            y0=self.N0,  # Starting conditions of the DGL
            t=t,
            args=(ext_func, s, t1, t2),
            Dfun=self.jac,  # The Jacobean of the DGL. Its optional.
            # The precisioin parameter for the nummerical DGL solver.
            rtol=self.rtol,
            atol=self.atol,
            full_output=self.full_output,
            **kwargs,
        )
        if self.full_output:
            ret, self.infodict = ret

        return ret

    def fit_populations(self, t):
        s, t1, t2, c1, mu = self.p
        return self.population(
            t,
            lambda t: self.ext_gaus(t, mu, self.gSigma),
            s,
            t1,
            t2
        )

    def start_population(self, t):
        s, t1, t2, c1, mu = self.p0
        return self.population(
            t,
            lambda t: self.ext_gaus(t, mu, self.gSigma),
            s,
            t1,
            t2
        )

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
            lambda t: self.ext_gaus(t, mu, self.gSigma),
            s,
            t1,
            t2
        ).T
        return ((N[0] + N[2] + c * N[3] - N[1])**2) / (self.N0[0]**2)

    def chi2(self, s, t1, t2, c, mu):
        """Chi2 to be minimized by minuit."""
        return np.sum(
            (
                (self.ydata - self.fit_func(self.xdata, s, t1, t2, c, mu)) /
                self.sigma
            )**2
        )

    def save(self, fname):
        """Save FourLevelMolKin results."""
        parameter_dict = {
           'gSigma': self.gSigma,
           'rtol': self.rtol,
           'atol': self.atol,
           'N0': self.N0,
        }
        super().__save__(fname, parameter_dict)


class FourLevelMolKin(CurveFitter, FourLevelMolkinBase):
    def __init__(self, *args, gSigma=150, N0=[1, 0, 0, 0],
                 rtol=1.09012e-9, atol=1.49012e-9, full_output=True, **kwargs):
        """Class for the four level Molecular Dynamics Model.

        *args and **kwargs are passed to CurveFitter

        Parameters
        ----------
        xdata: array
            The xdata of the model
        ydata: array
            The ydata of the model
        p0: Starting values for the model
        gSigma: width of the excitation pulse  in  the same
            units as xdata.
        N0: starting conditions for the Population Model.
        rtol, atol: precisioin of the numerical integrator.
            default if scipy is not enough. The lower
            this number the more exact is the solution
            but the slower the function.
        full_output: If true create infodict for odeint
            result is saved under self.infodict
        metadata:
            dictionary for metadata.
        """
        FourLevelMolkinBase.__init__(self, gSigma, N0, rtol, atol, full_output)
        CurveFitter.__init__(self, *args, **kwargs)


class FourLevelMolKinM(Minuitter, FourLevelMolkinBase):
    def __init__(
            self,
            xdata=None,
            ydata=None,
            p0=None,
            sigma=None,
            gSigma=150,
            N0=[1, 0, 0, 0],
            rtol=1.09012e-9,
            atol=1.49012e-9,
            full_output=True,
            fitarg={},
            **kwargs
    ):

        FourLevelMolkinBase.__init__(self, gSigma, N0, rtol, atol, full_output)
        Minuitter.__init__(self, xdata, ydata, p0, sigma, fitarg, **kwargs)

class ThreeLevelMolkin(Minuitter):
    def __init__(
            self,
            *args,
            gSigma=150,
            N0=[1, 0, 0],
            rtol=1.09012e-9,
            atol=1.49012e-9,
            full_output=True,
            **kwargs
    ):
        Minuitter.__init__(self, *args, **kwargs)
        self.gSigma = gSigma  # width of the excitation
        self.rtol = rtol  # Precition of the numerical integrator.
        self.atol = atol
        # Starting conditions of the Populations, not to be confuesed with starting conditions of the plot
        self.N0 = N0
        self.full_output = full_output
        self.infodict = None  # Infodict return of the Odeint.

    def ext_gaus(self, t, mu, sigma):
        """Gausian excitation function.

        Due to historic reasons its not a strict gausian, but something
        very cloe to it. The Igor Code is:
        1/sqrt(pi)/coeff1*exp(-(coeff0-x)^2/coeff1^2) """

        return 1 / np.sqrt(np.pi) / sigma * np.exp(-((mu-t)/sigma)**2)

    def dgl(self, N, t, ext_func, s, t1):
        """DGL of the three level system.

        Parameters
        ----------
        N: deg 3 Array with initial populations of the levels
            typically [1, 0, 0]
        t: float
             time
        ext_func: excitation function of the laser. Typically a gaussian.
            Function of t.
        s: scaling factor of the pump laser.
        t1: Livetime of the excited state.

        Returns
        -------
        Dericatives of the system as 3dim array.
        """
        A = np.array([
            [-s*ext_func(t), s*ext_func(t), 0],
            [s*ext_func(t), -s*ext_func(t) - 1/t1, 0],
            [0, 1/t1, 0]
        ], dtype=np.float64)
        dNdt = A.dot(N)
        return dNdt

    def population(self, t, ext_func, s, t1, **kwargs):
        """Nummerical solution of the DGL.

        Parameters
        ----------
        t: array if times
        ext_func: excitation function. Depends on t.
        s: scaling factor of the pump.
        t1: livetime of the first state.

        Returns
        -------
        Populations of the 3 levels at the times t.
        """

        ret = odeint(
            func=self.dgl,  # the DGL of the 3 level water system
            y0=self.N0,  # Starting conditions of the DGL
            t=t,
            args=(ext_func, s, t1),
            rtol=self.rtol,
            atol=self.atol,
            full_output=self.full_output,
            **kwargs,
        )
        if self.full_output:
            ret, self.infodict = ret

        return ret

    def fit_populations(self, t):
        s, t1, c1, mu = self.p
        return self.population(
            t,
            lambda t: self.ext_gaus(t, mu, self.gSigma),
            s,
            t1,
        )

    def fit_func(self, t, s, t1, c, mu):
        """
        Function we use to fit.

        parameters
        ----------
        t: time
        s: Gaussian Amplitude
        t1: Livetime of first state
        c: Coefficient of third(Heat) state
        scale: Scaling factor at the very end

        Returns:
        The bleach of the water model
        and the Matrix with the populations"""
        N = self.population(
            t,
            lambda t: self.ext_gaus(t, mu, self.gSigma),
            s,
            t1,
        ).T
        return ((N[0] + c * N[2] - N[1])**2) / (self.N0[0]**2)

    def chi2(self, s, t1, c, mu):
        """Chi2 to be minimized by minuit."""
        return np.sum(
            (
                (self.ydata - self.fit_func(self.xdata, s, t1, c, mu)) /
                self.sigma
            )**2
        )

class GaussianModel(CurveFitter):
    def __init__(self, *args, **kwargs):
        """Data class to describe gaussian shaped data.

        *args and **kwargs get passed to the CurveFitter class
        Parameters
        ----------
        xdata: array
        ydata: array
        p0: array of starting values, with
            [A, mu, sigma, c]
        metdata: dictionary with metdata.
        """
        super().__init__(*args, **kwargs)
        self._pnames = ("A", "mu", "sigma", "c")
        self._box_str_format = '{:5}: {:8.3g} $\\pm$ {:6.1g}\n'

    def fit_func(self, x, A, mu, sigma, c):
        """Guassian function

        A: amplitude
        mu: position
        sigma: std deviation
        c : offset
        """
        return A * norm.pdf(x, mu, sigma) + c
