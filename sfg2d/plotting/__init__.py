import matplotlib.pyplot as plt

from . import ts
from . import scan


def plot(*args, **kwargs):
    """Wrapper for default matplotlib.plot that adds xlabel as wavenumbers"""
    plt.plot(*args, **kwargs)
    plt.xlabel('Wavenumbers in cm$^{-1}$')
