import pandas as pd
import numpy as np
from .. import fit

class ScanBase():
    """ABS for Scans."""
    _med = None

    def __init__(self,
                 df=None, base=None, norm=None, metadata=None,
                 dbase=None, dnorm=None, normalized=None, dnormalized=None):
        self._df = df
        self._base = base
        self._dbase = dbase
        self._norm = norm
        self._dnorm = dnorm
        self._normalized = normalized
        self._dnormalized = dnormalized
        self.metadata = metadata
        self.fitarg = {}
        self.fit_roi = slice(None, None)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    @property
    def dnorm(self):
        return self._dnorm

    @dnorm.setter
    def dnorm(self, value):
        self._dnorm = value

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        if isinstance(value, np.ndarray):
            if len(np.shape(value.squeeze())) == 1:
                self.df['base'] = value.squeeze()
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            self.df['base'] = value
        self._base = value

    @property
    def dbase(self):
        return self._dbase

    @dbase.setter
    def dbase(self, value):
       self._dbase = value

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def sub_base(self, inplace=False, axis=0, **kwargs):
        """substract baseline from data"""
        ret = self._df.subtract(self._base, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    def add_base(self, inplace=False, axis=0, **kwargs):
        """add baseline to data"""
        ret = self._df.add(self._base, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    def normalize(self, axis=0, inplace=False, **kwargs):
        """normalize data"""
        ret = self._df.divide(self._norm, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    def un_normalize(self, axis=0, inplace=False, **kwargs):
        """unnormalize data """
        ret = self._df.multiply(self._norm, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    @property
    def normalized(self):
        if isinstance(self._normalized, type(None)):
            self._normalized = self._df.divide(self._norm, axis=0)
        return self._normalized

    @normalized.setter
    def normalized(self, value):
        self._normalized = value

    @property
    def dnormalized(self):
        if isinstance(self._dnormalized, type(None)):
            raise NotImplementedError
            #self._dnormalized =
        return self._dnormalized

    @dnormalized.setter
    def dnormalized(self, value):
        self._dnormalized = value

    def __str__(self):
        return self._df.__str__()

    def __repr__(self):
        return self._df.__repr__()

    @property
    def groupby_spec(self):
        return self._df.groupby(axis=1, level=0)

    @property
    def med(self):
        if isinstance(self._med, type(None)):
            self._med = self.groupby_spec.median()
        return self._med

    def make_chi2_fit(self, fit_func,
                      fitarg = None, y='normalized', x=None, error='dnormalized',
                      fit_roi = None, **kwargs):
        """add the results of a fit to the scan.

        Parameters
        ----------
        fit_func : function
        fitarg : dictionary
        x : string or array
            if none index of DataFrame is used
        y : string or array
        error : string or array
        fit_roi: slice
            selection of the data relecant for the fit
            if x, y or error is given as array, this is ignored.

        kwargs
        -------
        Are passed to sfg2d.fit.make_chi2_fit
        """
        if isinstance(fitarg, type(None)):
            fitarg = self.fitarg
        if isinstance(fit_roi, type(None)):
            fit_roi = self.fit_roi
        self.fit_roi = fit_roi
        if isinstance(x, type(None)):
            x = self.df[y].ix[fit_roi].index.get_values()
        if isinstance(x, str):
            x = self.df[x].ix[fit_roi].get_values()
        if isinstance(y, str):
            y = self.df[y].ix[fit_roi].get_values()
        if isinstance(error, str):
            error = self.df[error].ix[fit_roi].get_values()

        self.minuit, self.chi2 = fit.make_chi2_fit(
            x, y, fit_func, fitarg, error=error, **kwargs
        )

    def load_chi2_fit(self, fp, fit_func, **kwargs):
        """"""
        self.fitarg, self.fit_roi = fit.get_fitarg_from_savearg(
            fit.load_savearg(fp)
        )
        self.make_chi2_fit(
            fit_func,
            self.fitarg,
            **kwargs,
        )

class PumpProbe():
    """ABS for PumpProbe data"""
    _df = None
    _pumped = None
    _probed = None
    _pump = None
    pp_delays = None
    
    @property
    def pumped(self):
        return self._pumped

    @pumped.setter
    def pumped(self, value):
        if isinstance(self._df.get("bleach"), pd.core.series.Series):
            self.df.drop('bleach', axis=1, inplace=True)
        if isinstance(value, str):
            self._pumped = value
        else:
            raise NotImplementedError

    @property
    def probed(self):
        return self._probed

    @probed.setter
    def probed(self, value):
        if isinstance(self._df.get("bleach"), pd.core.series.Series):
            self.df.drop('bleach', axis=1, inplace=True)
        if isinstance(value, str):
            self._probed = value
        else:
            raise NotImplementedError

    @property
    def pump(self):
        return self._pump

    @pump.setter
    def pump(self, value):
        self._pump = value   

    
class Scan(ScanBase):
    def __init__(self, df=None, base=None, norm=None, metadata=None):
        self._df = df
        self._base = base
        self._norm = norm
        self.metadata = metadata

    def model_base(self):
        raise NotImplementedError


class TimeScan(ScanBase, PumpProbe):
    def __init__(self, df=None, base=None, norm=None, pump=None, metadata=None,
                 pumped="spec_0", probed="spec_1"):
        self._df = df
        self.metadata = metadata
        self._pump = pump
        self._base = base
        self._norm = norm
        self._pumped = pumped
        self._probed = probed

    def __getitem__(self, k):
        """ """
        if isinstance(k, int):
            return self._df.loc[self.pp_delays[k]]
        
        if isinstance(k, str):
            return self._df[k]

    def __deepcopy__(self, memodict={}):

        from copy import deepcopy
        return TimeScan(
            self._df.copy(),
            self._base,
            self._norm,
            self._pump,
            deepcopy(self.metadata),
            self._pumped,
            self._probed)

    def sub_base(self, inplace=False, axis=0, level=1, **kwargs):
        """substract baseline from data"""

        # When indeces are different pandas cand substract correctly
        # The result will then be onlay NaNs
        try:
            if any(self._df.index.levels[1] != self._base.index.levels[1]):
                level = None
        except AttributeError:
            pass
            
        ret = self._df.subtract(self._base, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    def add_base(self, inplace=False, axis=0, level=1, **kwargs):
        """add baseline to data"""
        ret = self._df.add(self._base, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
            self._med = None
            return
        return ret

    def normalize(self, axis=0, inplace=False, level=1, **kwargs):
        """normalize data"""
        ret = self._df.divide(self._norm, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
            # Normalization effects median and bleach and thus must
            # be removed on change of data
            self._med = None
            if isinstance(self._df.get("bleach"), type(self._df)):
                self._df.drop("bleach", axis=1, inplace=True)
            return
        return ret

    def un_normalize(self, axis=0, inplace=False, level=1, **kwargs):
        """unnormalize data """
        ret = self._df.multiply(self._norm, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
            # Normalization effects median and bleach and thus must
            # be removed on change of data
            self._med = None
            if isinstance(self._df.get("bleach"), type(self._df)):
                self._df.drop("bleach", axis=1, inplace=True)
            return
        return ret

    @property
    def pp_delays(self):
        # Cast to int needed because json.dump wont work otherwise
        return [int(elm) for elm in self._df.index.levels[0]]

    @property
    def groupby_pp_delay(self):
        return self._df.groupby(axis=0, level=0)

    @property
    def pumped(self):
        return self.med[self._pumped]

    @pumped.setter
    def pumped(self, value):
        if isinstance(self._df.get("bleach"), pd.core.series.Series):
            self.df.drop('bleach', axis=1, inplace=True)
        if isinstance(value, str):
            self._pumped = value
        else:
            raise NotImplementedError

    @property
    def probed(self):
        return self.med[self._probed]

    @probed.setter
    def probed(self, value):
        if isinstance(self._df.get("bleach"), pd.core.series.Series):
            self.df.drop('bleach', axis=1, inplace=True)
        if isinstance(value, str):
            self._probed = value
        else:
            raise NotImplementedError

    @property
    def bleach(self):
        if isinstance(self._df.get("bleach"), type(None)):
            self._df["bleach"] = self.pumped - self.probed
        return self._df["bleach"]

    @bleach.setter
    def bleach(self, value):
        self._df["bleach"] = value


