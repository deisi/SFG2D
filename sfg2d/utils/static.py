"""static functions go here """
from numpy import sqrt, power, cos, sin, arcsin, square, array, ones, shape, sum


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
