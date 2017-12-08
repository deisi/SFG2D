"""Fitting Models to Fit data with."""

import numpy as np
from scipy.integrate import odeint
from scipy.special import erf, erfc
from scipy.stats import norm, skewnorm
from iminuit import Minuit
import sys
thismodule = sys.modules[__name__]


def make_model_fit(
        model_name,
        xdata,
        ydata,
        yerr=None,
        fit=False,
        print_matrix=True,
        model_kwgs={}
):
    """Generig interface for model fits.
    **Arguments:**
      - **model_name**: String describing the name of the model
      - **xdata**: x-data to the model
      - **ydata**: y-data to model

    **Keywords:**
      - **yerr**: yerr to model
      - **fit**: boolean weather to run the fit
      - **model_kwgs**: Keywords passed to model during creation
    """

    model = getattr(thismodule, model_name)(xdata, ydata, yerr, **model_kwgs)
    if fit:
        fit_model(
            model, print_matrix=print_matrix
        )
    return model

def fit_model(model, minos=False, print_matrix=True):
    """Function to run migrad minimizations.

    **Arguments:**
      - **model**: Model instance to run migrad of.
    **Keywords:**
      - **minos**: Boolean, If Errors should be calculated with minos.
         Slow but more precise error estimation of the fit parameters.
    """
    model.minuit.migrad()  # Run the minimization
    if minos:
        model.minuit.minos()
        model.minuit.migrad()
    if print_matrix:
        model.minuit.print_matrix()


def normalize_trace(model, shift_mu=False, scale_amp=False, shift_heat=False):
    """Normalize trace.

    model: model to work on
    shift_mu: Schift by mu value of fit
    scale_amp: Scale by realtive height between heat and minimum.
    shift_heat: Make heat value equal

    returns shiftet data arrays with:
    """
    mu = 0
    if shift_mu:
        mu = model.minuit.fitarg['mu']

    offset = 0
    if shift_heat:
        offset=1-model.yfit_sample[-1]

    scale = 1
    if scale_amp:
        x_mask = np.where((model.xsample-mu>0) & (model.xsample-mu<1000))
        scale = 1-offset-model.yfit_sample[x_mask].min()

    xdata = model.xdata - mu
    ydata = (model.ydata+offset-1)/scale+1
    yerr = model.yerr/scale
    xsample = model.xsample - mu
    yfit_sample = (model.yfit_sample+offset-1)/scale+1
    return xdata, ydata, yerr, xsample, yfit_sample

class Fitter():
    def __init__(
            self,
            xdata=None,
            ydata=None,
            sigma=None,
            fitarg={},
            box_coords=None,
            roi=None,
            name='',
            ignore_errors=False,
            **kwargs
    ):
        """Base Class to fit with Minuit.

         - **ignore_errors**:
            Optional if given, sigmas will get ignored during the fit.
         **fitarg**: Dictionary gets passed to minuit.
          and sets the starting parameters.
        **kwargs:**
          Get passed to minuit. Most important is
        """
        self.xdata = xdata
        self.ydata = ydata
        self.sigma = sigma  # 1/y-errors.
        self.cov = None  # The covariance of the fit
        # Coordinates of the fit result box in fig coordinates
        self._box_coords = box_coords
        self._box_str_format = '{:2}: {:8.3g} $\pm$ {:6.1g}\n'
        if not roi:
            roi = slice(None)
        self.roi = roi
        self._pnames = None
        self._xsample_num = 400
        self.name = name
        self.ignore_errors=ignore_errors
        # Buffer for figures
        self.figures = {}
        kwargs.setdefault('pedantic', False)
        self.minuit = Minuit(self.chi2, **fitarg, **kwargs)

    @property
    def box_coords(self):
        if not self._box_coords:
            return self.xdata.mean(), self.ydata.mean()
        return self._box_coords

    def draw_text_box(self, box_coords=None, **kwargs):
        """Draw a textbox on current axes."""
        from matplotlib.pyplot import text
        if not box_coords:
            box_coords = self.box_coords
        text(*box_coords, self.box_str, **kwargs)


    @property
    def p(self):
        """Parameters of the Fit."""
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

    @property
    def xdata(self):
        """X data for the fit."""
        return self._xdata[self.roi]

    @xdata.setter
    def xdata(self, value):
        if len(np.shape(value)) != 1:
            raise IndexError('Shappe if xdata is not of dim 1')
        self._xdata = value

    @property
    def ydata(self):
        """Y data for the fit."""
        return self._ydata[self.roi]

    @ydata.setter
    def ydata(self, value):
        if len(np.shape(value)) != 1:
            raise IndexError('Shappe if xdata is not of dim 1')
        self._ydata = value

    @property
    def sigma(self):
        """Error of the ydata for the fit."""
        if isinstance(self._sigma, type(None)):
            return np.ones_like(self.ydata)
        if self.ignore_errors:
            return np.ones_like(self.ydata)
        return self._sigma[self.roi]

    @sigma.setter
    def sigma(self, value):
        if isinstance(value, type(None)):
            self._sigma = np.ones_like(self._ydata)
        elif len(np.shape(value)) != 1:
            raise IndexError('Shappe if xdata is not of dim 1')
        if np.any(value==0):
            from warnings import warn
            warn('Passed uncertainty has a 0 value\nIgnoring errorbars.\n{}'.format(value))
            self._sigma = value
            self.ignore_errors = True
        self._sigma = value

    @property
    def yerr(self):
        """Error of the ydata for the fit."""
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

    @property
    def xsample(self):
        return np.linspace(self.xdata[0], self.xdata[-1], self._xsample_num)

    @property
    def y_fit(self):
        return self.fit_res(self.xsample)

    @property
    def yfit_sample(self):
        return self.fit_res(self.xsample)


class GaussianModelM(Fitter):
    def __init__(self, *args, **kwargs):
        ''' Fit Gausian model using Minuit.
        **args**/**kwargs:**
          Get passed to `sfg2d.models.Fitter`. Options are:
            - **xdata**: array of x data points
            - **ydata**: array of y data points
            - **sigma**: Array of y data errors
            - **fitarg**: Dictionary with fit conditions.
                Each parameter has an entry with its name `'parameter'`
                `'error_parameter'` `'fix_parameter'` and `'limit_parameter'`
            - **box_coords**: Coordinates of the fit result box in data coordinates.
            - **roi**: Slice. Region of interest of the data.
              This subregion will be used for fitting.
            - **name**: Str, Name to describe the Model.
        '''
        Fitter.__init__(self, *args, **kwargs)
        self._box_str_format = '{:5}: {:7.3g} $\\pm$ {:6.1g}\n'

    def fit_func(self, x, A, mu, sigma, c):
        """Guassian function

        A: amplitude
        mu: position
        sigma: std deviation
        c : offset
        """
        return A * norm.pdf(x, mu, sigma) + c

    def chi2(self, A, mu, sigma, c):
        """Chi2 to be minimized by minuit."""
        return np.sum(
            (
                (self.ydata - self.fit_func(self.xdata, A, mu, sigma, c)) /
                self.sigma
            )**2
        )

class SkewedNormal(Fitter):
    def __init__(self, *args, **kwargs):
        Fitter.__init__(self, *args, **kwargs)
        self._box_str_format = '{:5}: {:7.3g} $\\pm$ {:6.1g}\n'

    def fit_funct(self, x, A, mu, sigma, kurt, c):
        return A * skewnorm.pdf(x, kurt, mu, sigma) + c


class FourLevelMolKinM(Fitter):
    def __init__(
            self,
            *args,
            gSigma=150,
            N0=[1, 0, 0, 0],
            rtol=1.09012e-9,
            atol=1.49012e-9,
            full_output=True,
            **kwargs
    ):
        """4 Level Model Fitter.

        To use set following `kwargs`
        `xdata`, `ydata` and `fitarg`. Optinal pass `sigma` for y errors.

        **Arguments:**
          - **N0**: Boundary condition of the DGL
          - **rtol**: Precision parameter of the DGL
          - **atol**: Precision parameter of the DGL
          - **full_output**: Weather to get full_output of the DGL Solver.
              Usefull for debugging. atol and rtol

        **args**/**kwargs:**
          Get passed to `sfg2d.models.Fitter`. Options are:
            - **xdata**: array of x data points
            - **ydata**: array of y data points
            - **sigma**: Array of y data errors
            - **fitarg**: Dictionary with fit conditions.
                Each parameter has an entry with its name `'parameter'`
                `'error_parameter'` `'fix_parameter'` and `'limit_parameter'`
            - **box_coords**: Coordinates of the fit result box in data coordinates.
            - **roi**: Slice. Region of interest of the data.
              This subregion will be used for fitting.
            - **name**: Str, Name to describe the Model.
        """

        self.gSigma = gSigma  # width of the excitation
        self.rtol = rtol  # Precition of the numerical integrator.
        self.atol = atol
        # Starting conditions of the Populations, not to be confuesed with starting conditions of the plot
        self.N0 = N0
        self.full_output = full_output
        self.infodict = None  # Infodict return of the Odeint.
        Fitter.__init__(self, *args, **kwargs)

    def ext_gaus(self, t, mu, sigma):
        """Gausian excitation function.

        Due to historic reasons its not a strict gausian, but something
        very cloe to it. The Igor Code is:
        1/sqrt(pi)/coeff1*exp(-(coeff0-x)^2/coeff1^2)

        The here wanted sigma is sqrt(2)*sigma of a normal gaussian
        and then its also normalized. If you have FWHM, then sigma
        is sigma = FWHM/(2*sqrt(log(2)))
        """

        return 1 / np.sqrt(np.pi) / sigma * np.exp(-((mu-t)/sigma)**2)

    # The Physical Water model
    def dgl(self, N, t, ext_func, s, t1, t2):
        """Dgl of the 4 Level DGL system.

        **Arguments:**
          - **N**: deg 4 array
              Population of the 4 levels respectively
          - **t**: float
              time
          - **ext_func**: exictation function in time.
              Time profile of the pump laser.
              Function of t. Usaully a gaussian function.
          - **s**: scaling factor of the pump laser.
          - **t1**: Time constant of first level
          - **t2**: Time constant of second level.

        **Returns:**
          Derivatives of the system. As 4 dim array.
        """

        # This is the DGL written as a Matrix multiplication.
        # dNdt = A x N
        # A is the constructing matrix of the DGL
        # and N is a 4-level vector with (N0, N1, N2, N3)
        # as the population of the states at time t.
        # dNdt is the state wise derivative of N
        # See https://en.wikipedia.org/wiki/Matrix_differential_equation

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

        **Arguments:**
          - **t**: array of time values
          - **ext_func**: Function of excitation.
          - **s**: scalar factor for the pump
          - **t1**: Live time of the first exited state
          - **t2**: livetime of the intermediate state.

        **Returns**
          (len(t), 4) shaped array with the 4 entires beeing the population
          of the N0 t0  N3 levels of the system
        """

        ret = odeint(
            func=self.dgl,  # the DGL of the 3 level water system
            y0=self.N0,  # Starting conditions of the DGL
            t=t,
            args=(ext_func, s, t1, t2),
            #Dfun=self.jac,  # The Jacobean of the DGL. Its optional.
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

        **Arguments:**
          - **t**: time
          - **s**: Gaussian Amplitude
          - **t1**: Livetime of first state
          - **t2**: livetime of second(intermediate) state
          - **c**: Coefficient of third(Heat) state


        **Returns**
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
        """Chi2 to be minimized by minuit.

        **Arguments:**
          - **s**: Gaussian Amplitude of the excitation
          - **t1**: Livetime of the first state
          - **t2**: Livetime of the second state
          - **c**: Coefficient of the heat

        """
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


class SimpleDecay(Fitter):
    def __init__(
            self,
            *args,
            xsample=None,
            xsample_ext=0.1,
            fit_func_dtype=np.float64,
            **kwargs
    ):
        """Fitting Model with convolution of single exponential and gaussian.

        **Arguments**:
          - **xsample**: Optional
              Stepping size of the convolution. Default minimal
              difference of xdata and in the range of xdata.
          - **xsample_ext**: Boundary effects of the convolution make int necesarry to,
              add additional Datapoints to the xsample data. By default 10% are
              added.
          - **fit_func_dtype**: The exponential function in the fitfunc can become
              very huge. To cope with that one can set the dtype of the fit func.

        **args**/**kwargs:**
          Get passed to `sfg2d.models.Fitter`. Options are:
            - **xdata**: array of x data points
            - **ydata**: array of y data points
            - **sigma**: Array of y data errors
            - **fitarg**: Dictionary with fit conditions.
                Each parameter has an entry with its name `'parameter'`
                `'error_parameter'` `'fix_parameter'` and `'limit_parameter'`
            - **box_coords**: Coordinates of the fit result box in data coordinates.
            - **roi**: Slice. Region of interest of the data.
              This subregion will be used for fitting.
            - **name**: Str, Name to describe the Model.
        """
        Fitter.__init__(self, *args, **kwargs)
        self._xsample = np.array([])

        self.xsample = xsample
        self.fit_func_dtype = fit_func_dtype
        self.xdata_step_size = np.diff(self.xdata).min()
        if not xsample:
            self.xsample = np.arange(
                self.xdata[0],
                self.xdata[-1],
                self.xdata_step_size
            )

    @property
    def xsample(self):
        return self._xsample

    @xsample.setter
    def xsample(self, value):
        self._xsample = value

    def fit_func(self, t, A, t1, c, mu, ofs, sigma):
        """Result of a convolution of Gausian an exponential recovery.

        This function is the Analytically solution to the convolution of:
        f = (- A*exp(-t/tau) + c)*UnitStep(t)
        g = Gausian(t, mu, sigma)
        result = Convolve(f, g)

        **Arguments:**
          - **t**: array of times
          - **A**: Amplitude of the recovery
          - **t1**: Livetime of the recovery
          - **c**: Convergence of the recovery
          - **mu**: Tempoaral Position of the Pulse
          - **ofs**: Global offset factor
          - **sigma**: Width if the gaussian
        """
        ## This dtype hack is needed because the exp cant get very large.
        return 1/2 * (
            c + c * erf((t - mu)/(np.sqrt(2) * sigma)) -
            A * np.exp(((sigma**2 - 2 * t * t1 + 2 * mu * t1)/(2 * t1**2)),
                       dtype=self.fit_func_dtype) *
            erfc((sigma**2 - t * t1 + mu * t1)/(np.sqrt(2) * sigma * t1))
        ) + ofs

    def chi2(self, A, t1, c, mu, ofs, sigma):
        """Chi2 to be minimized by minuit."""
        return np.sum(
            (
                (self.ydata - self.fit_func(
                    self.xdata, A, t1, c, mu, ofs, sigma
                )) / self.sigma
            )**2
        )


class ThreeLevelMolkin(Fitter):
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
        """

        **args**/**kwargs:**
          Get passed to `sfg2d.models.Fitter`. Options are:
            - **xdata**: array of x data points
            - **ydata**: array of y data points
            - **sigma**: Array of y data errors
            - **fitarg**: Dictionary with fit conditions.
                Each parameter has an entry with its name `'parameter'`
                `'error_parameter'` `'fix_parameter'` and `'limit_parameter'`
            - **box_coords**: Coordinates of the fit result box in data coordinates.
            - **roi**: Slice. Region of interest of the data.
              This subregion will be used for fitting.
            - **name**: Str, Name to describe the Model.
        """
        Fitter.__init__(self, *args, **kwargs)
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

