"""static functions go here """

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
    #return (wavenumbers_to_nm(x)*up_wl)/( wavenumbers_to_nm(x)+up_wl)
    return (1/(1/wavenumbers_to_nm(x) + 1/up_wl))

def savefig(filename, **kwargs):
    import matplotlib.pyplot as plt
    '''save figure as pgf, pdf and png'''
    plt.savefig('{}.pgf'.format(filename), **kwargs)
    plt.savefig('{}.pdf'.format(filename), **kwargs)
    plt.savefig('{}.png'.format(filename), **kwargs)
