"""Fitting Models to Fit data with."""

import numpy as np
from scipy.integrate import odeint
from scipy.special import erf, erfc
from scipy.stats import norm, skewnorm
from iminuit import Minuit, describe
import sys
import yaml
import logging
thismodule = sys.modules[__name__]

def read_fit_results(fname):
    with open(fname) as ifile:
        fit_results = yaml.load(ifile)
    return fit_results

def model_fit_record(
        model,
        record,
        kwargs_select_y,
        kwargs_select_x,
        kwargs_select_yerr=None,
        kwargs_model=None,
        run=False,
):
    """Make a model using selected data from SfgRecrod.

    **Parameters**:
      - **model**: String, Class name of the model to use
      - **record**: SfgRecord obj to select data from
      - **kwargs_ydata**: kwargs to select y data with
      - **kwargs_xdata**: kwargs to select x data with
      - **kwargs_model**: kwargs to pass to model
      - **kwargs_yerr**: kwargs to select yerr with

    **Keywords:**
      - **run**: Actually run the fit

    **Returns:**
    A model obj for the fit.
    """
    if not kwargs_model:
        kwargs_model = {}
    if not kwargs_select_yerr:
        raise NotImplementedError('Models without errorbar not implemented yet')
    logging.debug('Selecting model data from record with:')
    logging.debug('ydata :{}'.format(kwargs_select_y))
    xdata = record.select(**kwargs_select_x).squeeze()
    ydata = record.select(**kwargs_select_y).squeeze()
    yerr = record.sem(**kwargs_select_yerr).squeeze()

    logging.debug('Setting model with:')
    logging.debug('xdata: {}'.format(xdata))
    logging.debug('ydata: {}'.format(ydata))
    logging.debug('yerr: {}'.format(yerr))
    logging.debug('kwargs_module: {}'.format(kwargs_model))
    model = getattr(thismodule, model)(xdata, ydata, yerr, **kwargs_model)
    if run:
        fit_model(
            model, # print_matrix=print_matrix
        )

    return model


def make_model_fit(
        model,
        xdata,
        ydata,
        yerr=None,
        fit=False,
        print_matrix=True,
        model_kwargs={}
):
    """Generig interface for model fits.
    **Arguments:**
      - **model**: String name of model class
      - **xdata**: x-data to the model
      - **ydata**: y-data to model

    **Keywords:**
      - **yerr**: yerr to model
      - **fit**: boolean weather to run the fit
      - **model_kwargs**: Keywords passed to model during creation
    """

    model = getattr(thismodule, model)(xdata, ydata, yerr, **model_kwargs)
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
        try:
            model.minuit.print_matrix()
        except ValueError:
            pass


def normalize_trace(model, shift_mu=False, scale_amp=False, shift_heat=False, scale_x=None):
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
        offset = 1-model.yfit_sample[-1]

    scale = 1
    if scale_amp:
        x_mask = np.where((model.xsample-mu>0) & (model.xsample-mu<1000))
        scale = 1-offset-model.yfit_sample[x_mask].min()

    xdata = model.xdata - mu
    ydata = (model.ydata+offset-1)/scale+1
    yerr = model.yerr/scale
    xsample = model.xsample - mu
    yfit_sample = (model.yfit_sample+offset-1)/scale+1
    if scale_x:
        xdata = scale_x * xdata
        xsample = scale_x * xsample
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
        self.sigma = sigma  # y-errors.
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
        # Minuit is used for fitting. This makes self.chi2 the fit function
        #logging.info(self.chi2)
        kwargs['forced_parameters'] = self.parameters
        logging.info(fitarg)
        logging.info(kwargs)
        self.minuit = Minuit(self.chi2, **fitarg, **kwargs)

    def chi2(self, *args, **kwargs):
        """Sum of distance of data and fit. Weighted by uncertainty of data."""
        return np.sum(
            (
                (self.ydata - self.fit_func(self.xdata, *args, **kwargs)) /
                self.sigma
            )**2
        )

    def fit_func(self):
        """Fit function that must be implemented by child classes."""
        raise NotImplementedError

    @property
    def parameters(self):
        return describe(self.fit_func)[1:]

    @property
    def box_coords(self):
        """Coordinades for the fit results box."""
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
        #return self.minuit.args
        return [self.minuit.fitarg[param] for param in self.minuit.parameters]

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
            raise IndexError('Shappe if ydata is not of dim 1')
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
        self._sigma = value
        if isinstance(value, type(None)):
            self._sigma = np.ones_like(self._ydata)
        elif len(np.shape(value)) != 1:
            raise IndexError('Shappe of yerr is not of dim 1')
        if np.any(value==0):
            pos = np.where(value==0)
            #replace = np.nanmedian(value)
            logging.warn('Zero value within ucertainty.')
            logging.warn('Zero Values @ {}'.format(pos))
            #logging.warn('Replacing error with {}'.format(replace))
            #logging.warn('Errors passed were {}'.format(value))
            #self._sigma = np.ones_like(self._ydata)
            #self.ignore_errors = True

    @property
    def yerr(self):
        """Error of the ydata for the fit."""
        return np.array(self.sigma)

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
        """A sampled version of the xdata. `Fitter._xsample_num` is the number
        of samples.
        `Fitter.xsample` is used to generate a smooth plot of the fitting curve.
        """
        return np.linspace(self.xdata[0], self.xdata[-1], self._xsample_num)

    @property
    def ysample(self):
        """Y vales of the fit function sampled with `Fitter.xsample`."""
        return self.yfit_sample

    @property
    def y_fit(self):
        """Y vales of the fit function sampled with `Fitter.xsample`."""
        return self.yfit_sample

    @property
    def yfit_sample(self):
        """Y vales of the fit function sampled with `Fitter.xsample`."""
        return self.fit_res(self.xsample)

    @property
    def fitarg(self):
        """Minuit fitargs."""
        return self.minuit.fitarg

    def plot(self, kwargs_data=None, kwargs_fit=None):
        """Function to easily look at a plot. Not very flexible. But usefull during
        interactive sessions.
        """
        import matplotlib.pyplot as plt
        if not kwargs_data:
            kwargs_data = {}
        kwargs_data.setdefault('x', self.xdata)
        kwargs_data.setdefault('y', self.ydata)
        kwargs_data.setdefault('yerr', self.yerr)
        kwargs_data.setdefault('fmt', 'o')

        if not kwargs_fit:
            kwargs_fit = {}
        kwargs_fit.setdefault('x', self.xsample)
        kwargs_fit.setdefault('y', self.ysample)
        kwargs_fit.setdefault('color', 'r')

        plt.errorbar(**kwargs_data)
        plt.plot(**kwargs_fit)



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

        # Minuit passes negative values for sigma
        # and these values lead to failures of the fitting
        if sigma < 0:
            return 0
        return A * norm.pdf(x, mu, sigma) + c

    #def chi2(self, A, mu, sigma, c):
    #    """Chi2 to be minimized by minuit."""
    #    return np.sum(
    #        (
    #            (self.ydata - self.fit_func(self.xdata, A, mu, sigma, c)) /
    #            self.sigma
    #        )**2
    #    )

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

    def fit_func(self, t, s, t1, t2, c, mu):
        """
        Function we use to fit.

        **Arguments:**
          - **t**: time
          - **s**: Gaussian Amplitude
          - **t1**: Livetime of first state
          - **t2**: livetime of second(intermediate) state
          - **c**: Coefficient of third(Heat) state
          - **mu**: Position of pump pulse, the zero.


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
        return ((N[0] - N[1]+ N[2] + c * N[3])**2) / (self.N0[0]**2)

    def population(self, t, *args, **kwargs):
        """Numerical solution to the 4 Level DGL-Water system.

        **Arguments:**
          - **t**: array of time values
        **Args**:
          Arguments of the dgl function
          - **ext_func**: Function of excitation.
          - **s**: scalar factor for the pump
          - **t1**: Live time of the first exited state
          - **t2**: livetime of the intermediate state.

        **kwargs**:
          Get passe to differential equation solver odeing
        **Returns**
          (len(t), 4) shaped array with the 4 entires beeing the population
          of the N0 t0  N3 levels of the system
        """

        ret = odeint(
            func=self.dgl,  # the DGL of the 4 level water system
            y0=self.N0,  # Starting conditions of the DGL
            t=t,  # Time as parameter
            args=args,  # Aguments of the dgl
            # Dfun=self.jac,  # The Jacobean of the DGL. Its optional.
            # The precisioin parameter for the nummerical DGL solver.
            rtol=self.rtol,
            atol=self.atol,
            full_output=self.full_output,
            **kwargs,
        )
        if self.full_output:
            ret, self.infodict = ret

        return ret

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

    #def chi2(self, s, t1, t2, c, mu):
    #    """Chi2 to be minimized by minuit.

    #    **Arguments:**
    #      - **s**: Gaussian Amplitude of the excitation
    #      - **t1**: Livetime of the first state
    #      - **t2**: Livetime of the second state
    #      - **c**: Coefficient of the heat

    #    """
    #    return np.sum(
    #        (
    #            (self.ydata - self.fit_func(self.xdata, s, t1, t2, c, mu)) /
    #            self.sigma
    #        )**2
    #    )

    def save(self, fname):
        """Save FourLevelMolKin results."""
        parameter_dict = {
           'gSigma': self.gSigma,
           'rtol': self.rtol,
           'atol': self.atol,
           'N0': self.N0,
        }
        super().__save__(fname, parameter_dict)

#class FourLevelSolution(Fitter):
#    def __init__(
#            self,
#            *args,
#            N0=[1,0,0,0],
#            **kwargs
#    ):
#    """Model that uses the analytically solution of the 4 LevelSystem.
#
#    The Conzept for the solution was taken from: (doi:10.1021/jp003158e) Lock, A. J.; Woutersen, S. & Bakker, H. J.
#    """



class Crosspeak(FourLevelMolKinM):
    def __init__(
            self,
            *args,
            N0=[1, 0, 0, 0, 0],
            **kwargs
    ):
        """4 Level Model based crosspeak fitter.
        """
        FourLevelMolKinM.__init__(self, *args, N0=N0, **kwargs)

    def matrix(self, t, t1, teq, tup, tdown, ext_func, s):
        """Matrix to construct the DGL"""
        return np.array([
            [-s * ext_func(t), -s * ext_func(t), 0, 0, 0],
            [s * ext_func(t), -s * ext_func(t)-1/tup-1/t1, 1/tdown, 0, 0],
            [0, 1/tup, -1/tdown, 0, 0],
            [0, 1/t1, 0, -1/teq, 0],
            [0, 0, 0, 1/teq, 0]
        ], dtype=np.float64)


    def dgl(self, N, *args):
        """Matrix form of the DGL"""
        dNdt = self.matrix(*args).dot(N)
        return dNdt

    def fit_func(self, t, t1, teq, tup, tdown, mu, gSigma, s, c):
        """Function that is used for fitting the data.
        """
        N = self.population(
            t,
            t1,
            teq,
            tup,
            tdown,
            lambda t: self.ext_gaus(t, mu, gSigma),
            s,
        ).T
        # On Pump vibration
        y0 = (N[0] + c * N[3] - N[1])**2 / self.N0[0]**2
        # Off Pump vibration
        y1 = (N[0] + c * N[3] - N[2])**2 / self.N0[0]**2
        # Fit function is two dimensional because input data consist of two
        # traces.
        return np.array([y0, y1])

    @property
    def ydata(self):
        return self._ydata

    @ydata.setter
    def ydata(self, value):
        self._ydata = np.array(value)

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
        if np.any(value == 0):
            raise ValueError('Cant handle 0 errors')
            from warnings import warn
            warn('Passed uncertainty has a 0 value\nIgnoring errorbars.\n{}'.format(value))
            self._sigma = value
            self.ignore_errors = True
        self._sigma = value

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

    #def chi2(self, A, t1, c, mu, ofs, sigma):
    #    """Chi2 to be minimized by minuit."""
    #    return np.sum(
    #        (
    #            (self.ydata - self.fit_func(
    #                self.xdata, A, t1, c, mu, ofs, sigma
    #            )) / self.sigma
    #        )**2
    #    )


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

    #def chi2(self, s, t1, c, mu):
    #    """Chi2 to be minimized by minuit."""
    #    return np.sum(
    #        (
    #            (self.ydata - self.fit_func(self.xdata, s, t1, c, mu)) /
    #            self.sigma
    #        )**2
    #    )


class TwoExponentials(Fitter):
    def __init__(self, *args, **kwargs):
        """Two exponentials convoluted with gaussian. Dont use this. Its has
        a causality problem.
        """
        Fitter.__init__(self, *args, **kwargs)
        self.N0 = [1, 0, 0, 0]

    def fit_func(self, x, Amp, Aheat, t1, t2, offs, pwidth, mu):
        """Analytical solution to the four level system with gaussian excitation pulse."""
        e1 = np.exp((0.5*((t2**-2.)*((pwidth**2)+((-2.*(x*t2))+(2.*(mu*t2)))))))
        e2 = np.exp((0.5*((t1**-2.)*((pwidth**2)+((-2.*(x*t1))+(2.*(mu*t1)))))))
        er_1 = ((((2.**-0.5)*(((pwidth**2)+(mu*t2))-(x*t2)))/t2)/pwidth)
        er_2 = ((((2.**-0.5)*(((pwidth**2)+(mu*t1))-(x*t1)))/t1)/pwidth)
        er_3 = (((2.**-0.5)*(x-mu))/pwidth)

        aux0=(e1)*(erfc(er_1));
        aux1=(e2)*(erfc(er_2));
        aux2=Amp*(((offs+(offs*(erf(er_3))))-(Aheat*aux0))-aux1);

        output=0.5*aux2+1

        # Due to exp overflow, nans can occur.
        # However they result in 1 because they happen at negative times.
        output[np.isnan(output)] = 1
        # +1 to have right population
        return output


class FourLevel(Fitter):
    """Analytical Solution to the 4 Level Model."""
    def __init__(self, *args, **kwargs):
        Fitter.__init__(self, *args, **kwargs)
        self.N = 1 # Number of initial oszillators.

    def N(self, t, t1, t2, N10, N20=0, N30=0):
        """Populations of the solution to the 4 level model.
        This is only true for t>0.

        **Parameters:**
          - **t**: Time points to calculated population of
          - **t1**: Lifetime of first excited state
          - **t2**: Lifetime of intermediate (heat) state
          - **N10**: Fraction of initialy excited oszillators 0<N10<1
          - **N20**: Fraction of initialy excited oszillators in heated state
          - **N30**: Fraction if initialy excited oszillators in final state

        **Returns:**
          Tuple of N0, N1, N2, N3 at times t

        """
        N1 = np.exp(((-t)/t1))*N10
        aux0=(np.exp((((-t)/t2)-(t/t1))))*(((np.exp((t/t2)))-(np.exp((t/t1))))\
        *(N10*t2));
        N2=((np.exp(((-t)/t2)))*N20)+(aux0/(t1-t2));
        aux0=(((np.exp(((t/t1)+(t/t2))))*t1)+((np.exp((t/t1)))*t2))-((np.exp((\
        (t/t1)+(t/t2))))*t2);
        aux1=((np.exp((((-t)/t2)-(t/t1))))*(N10*(aux0-((np.exp((t/t2)))*t1))))\
        /(t1-t2);
        N3=((np.exp(((-t)/t2)))*((-1.+(np.exp((t/t2))))*N20))+(N30+aux1);

        N0 = self.N - N1 - N2 - N3
        return N0, N1, N2, N3

    def fit_func(self, t, Amp, t1, t2, c, mu, sigma):
        """Analytical solution to the 4 level system convoluted with gaussian"""
        pi=np.pi;
        #a0 = erf((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/sigma))
        def mysqrt(x): return np.sqrt(x)
        aux0=sigma*((t1**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux1=sigma*((t1**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux2=sigma*((t1**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux3=(((t1-t2)**2))*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma)))));
        aux4=sigma*(t1*(t2*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma)))))));
        aux5=sigma*(t1*(t2*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma)))))));
        aux6=sigma*(t1*(t2*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma)))))));
        aux7=sigma*((t2**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux8=sigma*((t2**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux9=sigma*((t2**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)/\
        sigma))))));
        aux10=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux11=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*((t1**2)*(-1.+(erf(aux10))))));
        aux12=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux13=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux12)))))));
        aux14=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux15=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux14)))))));
        aux16=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux17=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux16))))));
        aux18=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux19=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux18))))));
        aux20=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux21=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux20)))))));
        aux22=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux23=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux22)))))));
        aux24=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux25=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux24))))));
        aux26=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux27=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux26)))))));
        aux28=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux29=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux28))))));
        aux30=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux31=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux30))))));
        aux32=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux33=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux32))))));
        aux34=(0.5*((sigma**2)*(t1**-2.)))+((mu/t1)+((0.5*((sigma**2)*(t2**-2.\
           )))+((mu/t2)+(((sigma**2)/t2)/t1))));
        aux35=(((2.**-0.5)*mu)/sigma)+((((2.**-0.5)*sigma)/t1)+(((2.**-0.5)*\
        sigma)/t2));
        aux36=(mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf((aux35-(((2.**-0.5)*t)/sigma))))))));
        aux37=(0.5*((sigma**2)*(t1**-2.)))+((mu/t1)+((0.5*((sigma**2)*(t2**-2.\
           )))+((mu/t2)+(((sigma**2)/t2)/t1))));
        aux38=(((2.**-0.5)*mu)/sigma)+((((2.**-0.5)*sigma)/t1)+(((2.**-0.5)*\
        sigma)/t2));
        aux39=(mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf((aux38-(((2.**-0.5)*t)/sigma)))))));
        aux40=(0.5*((sigma**2)*(t1**-2.)))+((mu/t1)+((0.5*((sigma**2)*(t2**-2.\
           )))+((mu/t2)+(((sigma**2)/t2)/t1))));
        aux41=(((2.**-0.5)*mu)/sigma)+((((2.**-0.5)*sigma)/t1)+(((2.**-0.5)*\
        sigma)/t2));
        aux42=(mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf((aux41-(((2.**-0.5)*t)/sigma)))))));
        aux43=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t2))-(((2.**-0.5)\
        *t)/sigma);
        aux44=(np.exp(((2.*((sigma**2)*(t2**-2.)))+(((2.*mu)/t2)+((-2.*t)/t2))\
           )))*((mysqrt((2.*pi)))*(sigma*((t2**2)*(-1.+(erf(aux43))))));
        aux45=t1*(t2*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t1))-(t*t1)))/t1)/\
        sigma))));
        aux46=(np.exp((0.5*((t1**-2.)*((sigma**2)+((2.*(mu*t1))+(-2.*(t*t1))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux45));
        aux47=t1*(t2*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t1))-(t*t1)))/t1)/\
        sigma))));
        aux48=(np.exp((0.5*((t1**-2.)*((sigma**2)+((2.*(mu*t1))+(-2.*(t*t1))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux47));
        aux49=(t2**2)*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t1))-(t*t1)))/t1)/\
        sigma)));
        aux50=(np.exp((0.5*((t1**-2.)*((sigma**2)+((2.*(mu*t1))+(-2.*(t*t1))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux49));
        aux51=t1*(t2*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t2))-(t*t2)))/t2)/\
        sigma))));
        aux52=(np.exp((0.5*((t2**-2.)*((sigma**2)+((2.*(mu*t2))+(-2.*(t*t2))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux51));
        aux53=(t2**2)*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t2))-(t*t2)))/t2)/\
        sigma)));
        aux54=(np.exp((0.5*((t2**-2.)*((sigma**2)+((2.*(mu*t2))+(-2.*(t*t2))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux53));
        aux55=(3.*(Amp*aux46))+((Amp*(c*aux48))+((-2.*(Amp*aux50))+((Amp*(c*\
        aux52))+(Amp*aux54))));
        aux56=(-2.*((Amp**2)*(c*((np.exp(((aux40-(t/t2))-(t/t1))))*aux42))))+(\
        ((Amp**2)*(c*aux44))+aux55);
        aux57=((Amp**2)*((c**2)*((np.exp(((aux34-(t/t2))-(t/t1))))*aux36)))+((\
        2.*((Amp**2)*((np.exp(((aux37-(t/t2))-(t/t1))))*aux39)))+aux56);
        aux58=((Amp**2)*aux29)+((-2.*((Amp**2)*(c*aux31)))+(((Amp**2)*((c**2)*\
        aux33))+aux57));
        aux59=(2.*((Amp**2)*(c*aux23)))+((-2.*((Amp**2)*aux25))+((2.*((Amp**2)\
        *(c*aux27)))+aux58));
        aux60=(-2.*((Amp**2)*aux17))+((2.*((Amp**2)*(c*aux19)))+((2.*((Amp**2)\
        *aux21))+aux59));
        aux61=((Amp**2)*((c**2)*aux11))+((3.*((Amp**2)*aux13))+((-2.*((Amp**2)\
        *(c*aux15)))+aux60));
        aux62=((Amp**2)*((c**2)*((mysqrt((0.5*pi)))*aux8)))+((Amp*(c*((\
        mysqrt((2.*pi)))*aux9)))+aux61);
        aux63=(2.*((Amp**2)*(c*((mysqrt((2.*pi)))*aux6))))+(((Amp**2)*((\
        mysqrt((0.5*pi)))*aux7))+aux62);
        aux64=(2.*(Amp*((mysqrt((2.*pi)))*aux4)))+((-2.*(Amp*(c*((mysqrt((\
        2.*pi)))*aux5))))+aux63);
        aux65=(Amp*(c*((mysqrt((2.*pi)))*aux2)))+(((mysqrt((0.5*pi)))*(\
        sigma*aux3))+aux64);
        aux66=((Amp**2)*((mysqrt((0.5*pi)))*aux0))+(((Amp**2)*((c**2)*((\
        mysqrt((0.5*pi)))*aux1)))+aux65);
        aux67=(t2**2)*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t2))-(t*t2)))/t2)/\
        sigma)));
        aux68=(np.exp((0.5*((t2**-2.)*((sigma**2)+((2.*(mu*t2))+(-2.*(t*t2))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux67));
        aux69=t1*(t2*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t2))-(t*t2)))/t2)/\
        sigma))));
        aux70=(np.exp((0.5*((t2**-2.)*((sigma**2)+((2.*(mu*t2))+(-2.*(t*t2))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux69));
        aux71=(t1**2)*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t1))-(t*t1)))/t1)/\
        sigma)));
        aux72=(np.exp((0.5*((t1**-2.)*((sigma**2)+((2.*(mu*t1))+(-2.*(t*t1))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux71));
        aux73=(t1**2)*(erfc(((((2.**-0.5)*(((sigma**2)+(mu*t1))-(t*t1)))/t1)/\
        sigma)));
        aux74=(np.exp((0.5*((t1**-2.)*((sigma**2)+((2.*(mu*t1))+(-2.*(t*t1))))\
           ))))*((mysqrt((2.*pi)))*(sigma*aux73));
        aux75=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t2))-(((2.**-0.5)\
        *t)/sigma);
        aux76=(np.exp(((2.*((sigma**2)*(t2**-2.)))+(((2.*mu)/t2)+((-2.*t)/t2))\
           )))*((mysqrt((0.5*pi)))*(sigma*((t2**2)*(-1.+(erf(aux75))))));
        aux77=((((aux66-(Amp*(c*aux68)))-(Amp*aux70))-(Amp*(c*aux72)))-(Amp*\
        aux74))-((Amp**2)*((c**2)*aux76));
        aux78=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t2))-(((2.**-0.5)\
        *t)/sigma);
        aux79=(np.exp(((2.*((sigma**2)*(t2**-2.)))+(((2.*mu)/t2)+((-2.*t)/t2))\
           )))*((mysqrt((0.5*pi)))*(sigma*((t2**2)*(-1.+(erf(aux78))))));
        aux80=(0.5*((sigma**2)*(t1**-2.)))+((mu/t1)+((0.5*((sigma**2)*(t2**-2.)))+((mu/t2)+(((sigma**2)/t2)/t1))));
        aux81=(((2.**-0.5)*mu)/sigma)+((((2.**-0.5)*sigma)/t1)+(((2.**-0.5)*\
        sigma)/t2));
        aux82=(mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf((aux81-(((2.**-0.5)*t)/sigma))))))));
        aux83=(aux77-((Amp**2)*aux79))-((Amp**2)*((np.exp(((aux80-(t/t2))-(t/\
        t1))))*aux82));
        aux84=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux85=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux84)))))));
        aux86=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t2))-(((2.**-0.5)*\
        t)/sigma);
        aux87=(np.exp((((0.5*((sigma**2)*(t2**-2.)))+(mu/t2))-(t/t2))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux86)))))));
        aux88=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux89=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((2.*pi)))*(sigma*((t1**2)*(-1.+(erf(aux88))))));
        aux90=((aux83-((Amp**2)*((c**2)*aux85)))-((Amp**2)*aux87))-((Amp**2)*(\
        c*aux89));
        aux91=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux92=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((0.5*pi)))*(sigma*((t1**2)*(-1.+(erf(aux91))))));
        aux93=((((2.**-0.5)*mu)/sigma)+(((mysqrt(2.))*sigma)/t1))-(((2.**-0.5)\
        *t)/sigma);
        aux94=(np.exp(((2.*((sigma**2)*(t1**-2.)))+(((2.*mu)/t1)+((-2.*t)/t1))\
           )))*((mysqrt((0.5*pi)))*(sigma*((t1**2)*(-1.+(erf(aux93))))));
        aux95=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux96=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*(t1*(t2*(-1.+(erf(aux95)))))));
        aux97=((aux90-((Amp**2)*((c**2)*aux92)))-((Amp**2)*aux94))-((Amp**2)*(\
        (c**2)*aux96));
        aux98=((((2.**-0.5)*mu)/sigma)+(((2.**-0.5)*sigma)/t1))-(((2.**-0.5)*\
        t)/sigma);
        aux99=(np.exp((((0.5*((sigma**2)*(t1**-2.)))+(mu/t1))-(t/t1))))*((\
        mysqrt((2.*pi)))*(sigma*((t1**2)*(-1.+(erf(aux98))))));
        aux100=sigma*((t2**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*\
        t)/sigma))))));
        aux101=sigma*((t2**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*\
        t)/sigma))))));
        aux102=((aux97-((Amp**2)*aux99))-((Amp**2)*(c*((mysqrt((2.*pi)))*\
        aux100))))-(Amp*((mysqrt((2.*pi)))*aux101));
        aux103=sigma*(t1*(t2*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)\
        /sigma)))))));
        aux104=sigma*(t1*(t2*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*t)\
        /sigma)))))));
        aux105=(aux102-((Amp**2)*((c**2)*((mysqrt((2.*pi)))*aux103))))-((\
        Amp**2)*((mysqrt((2.*pi)))*aux104));
        aux106=sigma*((t1**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*\
        t)/sigma))))));
        aux107=sigma*((t1**2)*(1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*\
        t)/sigma))))));
        aux108=(aux105-((Amp**2)*(c*((mysqrt((2.*pi)))*aux106))))-(Amp*((\
        mysqrt((2.*pi)))*aux107));
        aux109=(((t1-t2)**2))*(-1.-(erf(((((2.**-0.5)*mu)/sigma)-(((2.**-0.5)*\
        t)/sigma)))));
        aux110=((2.*pi)**-0.5)*(((t1-t2)**-2.)*(aux108-((mysqrt((0.5*pi)\
           ))*(sigma*aux109))));
        output=aux110/sigma;
        return output
