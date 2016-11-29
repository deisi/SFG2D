cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos

cdef complex chi_resonant_pure(double x, double amplitude, double pos, double width):
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
    cdef double delta
    cdef double CHiR_i
    cdef double ChiR_r
    cdef complex ChiR
    cdef double A
    cdef double gamma

    A = amplitude
    gamma = width / 2
    delta = pos - x
    ChiR_i = A * gamma / (delta**2 + gamma**2)
    ChiR_r = A * delta / (delta**2 + gamma**2)
    ChiR = ChiR_r + 1j * ChiR_i

    return ChiR

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef complex chi_resonant_multi_pure(double x, np.ndarray[np.double_t] res_args):
    """Multiple resonances.

    Parameters
    ---------
    res_args : array
        The length of thre res_args array determines the number of resonances.
        For each 3 values a new resonance is created.
    """
    cdef unsigned int n_res = res_args.shape[0] // 3
    cdef unsigned int i
    cdef double amplitude
    cdef double pos
    cdef double width
    cdef complex ChiRs = 0
    for i from 0 <= i < n_res:
        amplitude = res_args[3*i+0]
        pos = res_args[3*i+1]
        width = res_args[3*i+2]
        ChiRs += chi_resonant_pure(x, amplitude, pos, width)
    return ChiRs

cdef double sfgn_pure(x, double nr, double phase, np.ndarray[np.double_t, ndim=1] res_args):
    """
    Parameters
    ----------
    x : double
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
    cdef complex Chi

    # Non resonant part
    cdef complex ChiNR = nr * (cos(phase) + 1j * sin(phase))

    # Resonant part
    cdef complex ChiR = chi_resonant_multi_pure(x, res_args)

    #The physical Chi2
    Chi = ChiR + ChiNR
    return Chi.real ** 2 + Chi.imag ** 2

# Vectoriced version of the former function
# Sadly c doesn't support overloading and and
# cython with cpp I don't understand yet.
cdef np.ndarray[np.double_t] sfgn_pure_v(np.ndarray[np.double_t] x, double nr, double phase, np.ndarray[np.double_t] res_args):
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
    cdef unsigned int x_len = x.shape[0]
    cdef unsigned int i = 0
    cdef double xi
    cdef np.ndarray[np.double_t] Chi = np.empty(x_len)
    cdef complex Chii

    # Non resonant part
    cdef complex ChiNR = nr * (cos(phase) + 1j * sin(phase))

    for i from 0 <= i < x_len:
        xi = x[i]
        Chii = ChiNR + chi_resonant_multi_pure(xi, res_args)
        Chi[i] = Chii.real **2 + Chii.imag ** 2

    return Chi

#########################################
# Here the function that a accessible via python
#########################################

@cython.embedsignature(True)
cpdef double sfg2r_pure(double x, double nr, double phase,
         double amplitude, double pos, double width,
         double amplitude1, double pos1, double width1):
    """2 peak sfg response function.

    The function is written using cython as c backend. This is roughly 10x
    faster then the native c code.

    Parameters:
    x : double
    nr : double
        Non Resonatn background offset
    phase: double
        Phase of the non resonant background
    amplitude : double
    pos : double
    width : double
    amplitude1: double
    pos1 : double
    width1 : double.
    """

    cdef double ret = sfgn_pure(
        x,
        nr,
        phase,
        np.array([amplitude, pos, width, amplitude1, pos1, width1], dtype=np.double)
    )
    return ret

@cython.embedsignature(True)
cpdef np.ndarray[np.double_t] sfg2r_pure_v(np.ndarray[np.double_t] x, double nr, double phase,
                        double amplitude, double pos, double width,
                        double amplitude1, double pos1, double width1):
    """2 peak sfg response function.

    The function is written using cython as c backend. This is roughly 10x
    faster then the native c code.

    Parameters:
    x : array
    nr : double
        Non Resonatn background offset
    phase: double
        Phase of the non resonant background
    amplitude : double
    pos : double
    width : double
    amplitude1: double
    pos1 : double
    width1 : double.
    """

    cdef np.ndarray[np.double_t] ret = sfgn_pure_v(
        x,
        nr,
        phase,
        np.array([amplitude, pos, width, amplitude1, pos1, width1], dtype=np.double)
    )
    return ret
