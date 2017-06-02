"""static functions go here """
from os import path
from scipy.stats import norm
from numpy import (sqrt, power, cos, sin, arcsin, square, array, abs,
                   zeros_like, sum, argmax, argmin, e, where, resize, shape,
                   zeros, exp, convolve, where, ndarray, all)
from scipy.interpolate import interp1d
from .consts import STEPSIZE, XE, XG


def wavenumbers_to_nm(wavenumbers):
    """wavenumbers from given nm"""
    return 10**7/wavenumbers


def nm_to_wavenumbers(nm):
    """calculate wavenumbers from given nm"""
    return 10**7/nm


def nm_to_ir_wavenumbers(x, up_wl):
    """nm to vibrational wavenumbers

    The wavelength of the upconvertion photon is subtracted so that only
    the vibrational part of the signal in kept.

    Parameters
    ----------
    up_wl : int
        wavelength of the upconvertion pulse in nm"""
    return nm_to_wavenumbers(1/(1/x - 1/up_wl))

def ir_wavenumbers_to_nm(x, up_wl):
    """Translate ir wavenumbers to upconverted nm.

    The wavelength of the vibrational ir signal is upconverted, so that
    the real detected wavelength in nm is returned

    Parameters
    ----------
    x : array like
        ir wavenumbers to convert to nm wavelength
    up_wl : int
        wavelength of the up-conversion photon in nm.

    Returns
    -------
    float or array

    """
    return (1/(1/wavenumbers_to_nm(x) + 1/up_wl))

def get_interval_index(input_array, min, max):
    """Helper function to get index positioins from an sorted input_array.

    Parameters
    ----------
    input_array: sorted iterable
         array to search in
    min: int
        lower wavenumber boundary
    max: int
        uppper wavenumber boundary"""
    if input_array[0] < input_array[-1]:
        return argmax(input_array > min), argmin(input_array < max)
    else:
        return argmax(input_array < max), argmin(input_array > min)

def find_nearest_index(input_array, points):
    """Find the indices where input_array and points are closest to each other.

    input_array: array to search for indeced indeced
    points: array of values to find the closest index of.

    Returns:
    list of indices, that are closest to the given points throughout
    input_array."""
    points = resize(points, (shape(input_array)[0], shape(points)[0])).T
    wavenumbers = resize(input_array, points.shape)
    ret = abs(wavenumbers - points).argmin(1)
    return ret

def savefig(filename, dpi=150, pgf=False, **kwargs):
    import matplotlib.pyplot as plt
    '''save figure as pdf and png'''
    if pgf:
        plt.savefig('{}.pgf'.format(filename), **kwargs)
    plt.savefig('{}.pdf'.format(filename), **kwargs)
    plt.savefig('{}.png'.format(filename), dpi=dpi, **kwargs)
    print("Saved figure to: {}".format(path.abspath(filename)))


def Rs(ca, cb, n1, n2):
    """
    Refraction coefficient

    Neglects absorption.
    Parameters
    ----------
    ca: cos alpha
    cb: cos beta
    n1: refective index of medium 1
    n2: refrective index of medium 2

    Returns
    -------
    Reflection coefficient
    """
    return ((n1*ca - n2*cb)/(n1*ca + n2*cb))**2


def Ts(ca, cb, n1, n2):
    """Transmission coefficient.

    Neglects absorption.

    Parameters
    ----------
    ca: cos alpha
    cb: cos beta
    n1: refective index of medium 1
    n2: refrective index of medium 2

    Returns
    -------
    Transmission coefficient
    """
    return 1-Rs(ca, cb, n1, n2)


def n_caf2(x):
    '''Refractive index of CaF2.

    This is a numerical expression for the dispersion relation of CaF2

    taken from: http://refractiveindex.info/?shelf=main&book=CaF2&page=Malitson

    Parameters
    ----------
    x : array
        wavelength in \mu m

    Returns
    -------
    array of refractive index values
    '''
    return sqrt(
        1 +
        0.5675888/(1-power(0.050263605/x, 2)) +
        0.4710914/(1-power(0.1003909/x, 2)) +
        3.8484723/(1-power(34.649040/x, 2))
    )


def Rs_CaF(wavelength, alpha):
    """Reflection coeficient for Air <> CaF2 interface.

    Neglects absorption and uses Snelius for the Transmitted beam.

    Parameters
    ----------
    wavelength: mu m wavelength

    alpha: angle of incidence in rad

    Returns
    -------
    Reflektion coefficient"""

    n1 = 1
    n2 = n_caf2(wavelength)
    ca = cos(alpha)
    cb = cos(arcsin(n1*sin(alpha)/n2))
    Rs_CaF = Rs(ca, cb, n1, n2)

    return Rs_CaF

def chi_non_resonant(amplitude, phase):
    """Non Resonant Chi2 response.

    Parameters
    ----------
    amplitude: float
        The amplitude of the non resonant background

    phase: float
        The phase of the non resonant background

    Returns
    -------
    float: The non resonant background
    """
    ChiNR = amplitude * (cos(phase) + 1j * sin(phase))
    return ChiNR

def chi_resonant(x, amplitude, pos, width):
    """lorenzian chi resonance.

    Parameters
    ----------
    x : np.array
        The x axis, wavenumbers of frequencies
    amplitude:
        The amplitude of the resonance
    pos:
        The position of the resonance
    width:
        The FWHM of the resonance
    """
    A = amplitude
    delta = pos - x
    gamma = width / 2
    ChiR_i = A * gamma / (delta**2 + gamma**2)
    ChiR_r = A * delta / (delta**2 + gamma**2)
    ChiR = ChiR_r + 1j * ChiR_i
    return ChiR

def chi_resonant_multi(x, res_args):
    """Multiple resonances.

    Parameters
    ---------
    res_args : array
        The length of thre res_args array determines the number of resonances.
        For each 3 values a new resonance is created.
    """
    #print(x)
    number_of_resonances = len(res_args)//3
    # List of Results per resonance
    ChiRs = []
    for i in range(len(res_args)//3):
        amplitude, pos, width = [ res_args[3*i+j] for j in range(3)]
        ChiRs.append(chi_resonant(x, amplitude, pos, width))
    # The sum makes the superposition of the resonances
    return sum(ChiRs, 0)


def sfgn1(x, nr, phase, amplitude, pos, width):
    '''One resonance sfg response with NR background

    Parameters
    ----------
    x : array
        wavenumbers
    nr : Non Resonant background (amplitude)
    phase : Phase of the non resonant background
    amplitude : number
        Amplitude
    pos : number
    width : width of the lorenzian (FWHM)

    Returns
    -------
    array with the same shape as x of results
    '''

    # Non resonant part
    ChiNR = chi_non_resonant(nr, phase)

    # Resonant part
    ChiR = chi_resonant_multi(x, [amplitude, pos, width])

    # The physical Chi
    Chi = ChiR + ChiNR

    # Doing it this way seems to be the fastest
    return square(Chi.real) + square(Chi.imag)

def sfgn(x, nr, phase, *res_args):
    """
    Parameters
    ----------
    x : np.array
        x-data
    nr:
        Non resonant amplitude
    phase:
        Non resonant phase
    res_args: n*3 arguments
        Must have n*3 elements. If longer tailing elements are dropped
        List of args for the resonant peaks. The number of args
        determines the number of resonances. It is given by number of
        args divided by 3.

    """

    # Non resonant part
    ChiNR = nr * (cos(phase) + 1j * sin(phase))

    # Resonant part
    ChiR = chi_resonant_multi(x, res_args)

    #The physical Chi2
    # All resonant Chis are superpositioned, thus .sum(0)
    Chi = ChiR + ChiNR
    return square(Chi.real) + square(Chi.imag)

def sfg2r(x, nr, phase,
         amplitude, pos, width,
         amplitude1, pos1, width1):
    ret = sfgn(
            x, nr, phase,
            amplitude, pos, width,
            amplitude1, pos1, width1
          )
    return ret

def heat_time(t, H0, tau=1000, c=0):
    """Function to model the time dependecy of the heat
    ----------------
    Parameters
    t:        array type
        t time points

    H0:        number
        Amlitude or max heat of the model

    tau:     number
        time constant of the heat model

    c:        number
        time offset of the model

    ----------------
    return
        array of resulting values
    """
    HM = lambda x: H0*(1-e**(-1*x/tau))+c
    if hasattr(t, '__iter__'):
        ret = array([HM(time) for time in t])
        # need to remove negative times, because
        # model is unphysical in this region
        mask = where(t <= 0)
        ret[mask] = 0
        return ret
    if t <= 0:
        return 0
    return HM(t)

def gaus_func(x, A=1, mu=0, sigma=1, c=0):
    """Gaussian function

    Parameters
    ----------
    x : array
        position of interest
    A : num
        amplitude
    mu : num
        position
    simga : num
        width
    c : num
        offset

    Returns
    -------
    array
        points on a gausian distribution at point(s) x"""
    return A * norm.pdf(x, loc=mu, scale=sigma) + c

def exp_func(x, A=1,  tau=1, c=0):
    """Function of the exponential decay

    Parameters
    ----------
    x : array
        position(s) of interest
    A : num
        amplitude
    tau : num
        lifetime
    c : num
        offset parameter

    Returns
    -------
    array
        function at requested x points
    """
    if not isinstance(x, ndarray):
        x = array([x])
    ret = zeros_like(x, dtype='float64')
    if all(x <= 0):
        return ret
    mask = where(x > 0)
    ret[mask] = A*exp(-x[mask]/tau) + c
    return ret

def conv_gaus_exp(Xe, Xg, A0, A1, tau0, tau1, c, Ag=-1, sigma=0.25, mode="same", **kwargs):
    """
    Convolution of gaus and exp decay

    Parameters
    ----------
    A : num
        Amplitude of the exp functions
    tau : num
        lifetime of the decay in units of Xe
    c : num
        Offsetparameter of exp decay

    Ag: Amplitude of the gaussian

    sigma: width of the gaussian in units of Xg

    sigma: width of the gaussion in units of XE and XG
        I suggest ps, because numbers work best there.

    mode: mode of the convolution.
    kwargs are passed to numpy.signal.convolve

    Returns
    -------
    array
       negative and normalized version of the convolution of a gaussian
        and an exponential decay"""
    res = convolve(
        exp_func(Xe, A0, tau0, 0) +
        exp_func(Xe, A1, tau1, c),
        gaus_func(Xg, Ag, 0, sigma, 0),
        mode=mode,
        **kwargs,
    )
    return  res

def conv_gaus_exp_f(Xe, Xg, A0, A1, tau0, tau1, c, Ag=-1, sigma=0.25, mode="same", normalized=True, **kwargs):
    """Functioniced and vectorized form of the convolution

    can be used as: *sfg2d.double_decay*
    Parameters
    ----------
    Interpolated Version of the gaus convoluted double decay function.

    Xe: Sampling array for the decay functions.
        I recommend ps, because values ~1 work best in nummerical functions.

    Xg: Sampling of the gaussian.
        Should be a lot smaller then Xe but with the same stepsize.

    normalized: Normalized result such that Minimal value = -1

    for the rest,
    Rest see *sfg2d.utisl.static.conv_gaus_exp*

    Returns
    -------
    interp1d object
        a function that is defined throughout the Xe array."""
    conv = conv_gaus_exp(Xe, Xg, A0, A1, tau0, tau1, c, Ag, sigma, mode, **kwargs)
    if normalized:
        conv /= abs(conv).max()
    # TODO This should pick X according to mode and Xe and Xg
    if mode is "same":
        conv = interp1d(Xe, conv)  # interpolation vectorizes result
    else:
        raise IOError("not Implemented")

    return conv
