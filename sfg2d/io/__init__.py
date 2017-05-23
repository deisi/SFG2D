from . import veronica, victor_controller
from .spe import PrincetonSPEFile3
from .ntb import NtbFile

def load_fitarg(fp):
    """load the fit results from fp"""
    from json import load

    with open(fp) as json_data:
        ret = load(json_data)
    return ret

def load_fitarg_and_minuit(fp, fit_func, migrad=True):
    """Load fit results from fp and add to scan.

    Parameters
    ----------
    fp : string
        filepath
    fit_func : The function of the fitarg
    migrad : Boolean
        If true the fit will be performed again to offer additional properties

    Returns
    -------
    fitarg : dictionary
    minuit : Minuit obj with the fit
    """
    from iminuit import Minuit

    fitarg = load_fitarg(fp)
    minuit = Minuit(fit_func, **fitarg, pedantic=False)
    if not migrad:
        return fitarg, minuit
    print('**********************************************************************')
    print('Fitting with values from %s' % fp)
    minuit.migrad()
    return fitarg, minuit

def load_fitarg_minuit_chi2(fp, fit_func, x, y, migrad=True, **kwargs):
    """Load fitresult and construct minuit and chi2 from it.

    Parameters
    ----------
    fp : str
        path to fitarg file
    fit_func : function
        The function the fit is performed with

    x : array
        x-data
    y : array
        y-data

    kwargs
    ------
    migrad : Boolean
        The fit is directly performed
    Other kwargs are passed to the Chi2Regression.
    Most important one is
    **y_err : The y_err of the fit
    For the rest see probfit.Chi2Regression

    Returns
    -------
    fitarg: dictionary
        Fit results
    minuit : Minuit ob of the fit
    chi2 : Probfit.Chi2Regression
        Probfit obj. to determine the quality of the fit.
        This makes only sense, when at least an y_err was provided. Otherwise this is
        a useless thing
    """
    from probfit import Chi2Regression

    fitarg, minuit = load_fitarg_and_minuit(fp, fit_func, migrad)
    chi2 = Chi2Regression(
        fit_func,
        x,
        y,
        **kwargs
    )
    return fitarg, minuit, chi2
