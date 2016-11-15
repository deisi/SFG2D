"""static functions go here """
from numpy import sqrt, power, cos, sin, arcsin, square


def wavenumbers_to_nm(wavenumbers):
    """wavenumbers from given nm"""
    return 10**7/wavenumbers


def nm_to_wavenumbers(nm):
    """calculate wavenumbers from given nm"""
    return 10**7/nm


def nm_to_ir_wavenumbers(x, up_wl):
    """nm to vibrational wavenumbers

    The wavelength of the upconvertion photon is subtrackted so that only
    the vibrational part of the signal in kept.

    Parameters
    ----------
    up_wl : int
        wavelength of the upconvertion pulse in nm"""
    return nm_to_wavenumbers(1/(1/x - 1/up_wl))


def ir_wavenumbers_to_nm(x, up_wl):
    """ir wavenumbers to upconverted nm 

    The wavelength of the vibrational ir signal is upconverted, so that
    the real detected wavelength in nm is returned

    Parameters
    ----------
    x : array linke
        ir wavenumbers to convert to nm wavelength
    up_wl : int
        wavelength of the upconversion photon in nm.

    Returns
    -------
    float or array

    """
    return (1/(1/wavenumbers_to_nm(x) + 1/up_wl))


def savefig(filename, **kwargs):
    import matplotlib.pyplot as plt
    '''save figure as pgf, pdf and png'''
    plt.savefig('{}.pgf'.format(filename), **kwargs)
    plt.savefig('{}.pdf'.format(filename), **kwargs)
    plt.savefig('{}.png'.format(filename), **kwargs)


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
    """
    Transmission coefficient

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

    This is a nummerical expresion for the dispersion relation of CaF2

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

def sfgn1(x, nr, phase, amplitude, pos, width):
    '''One resonance sfg responde with NR background

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
    ChiNR = nr * (cos(phase) + 1j * sin(phase))

    # Resonent part
    A = amplitude
    delta = pos - x
    gamma = width / 2
    # probfit uses a precompiled c backend and is thus faster
    # ChiR_r = A * probfit.pdf.cauchy(x, pos, gamma)
    # ChiR_i = A * gamma * probfit.pdf.cauchy(x, pos, gamma)
    ChiR_i = A * gamma / (delta**2 + gamma**2)
    ChiR_r = A * delta / (delta**2 + gamma**2)
    ChiR = ChiR_r + 1j * ChiR_i

    # The physical Chi
    Chi = ChiR + ChiNR

    # Doing it this way seems to be the fastest
    return square(Chi.real) + square(Chi.imag)
