import copy # this really should be here? Seems like something is not right

class ScanBase():
    """ABS for Scans."""
    
    _df = None
    _base = None
    _norm = None
    _med = None
    metadata = None

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        self._base = value

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
            return
        return ret

    def add_base(self, inplace=False, axis=0,**kwargs):
        """add baseline to data"""
        ret = self._df.add(self._base, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            return
        return ret

    def normalize(self, axis=0, inplace=False, **kwargs):
        """normalize data"""
        ret = self._df.divide(self._norm, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            return
        return ret

    def un_normalize(self, axis=0, inplace=False, **kwargs):
        """unnormalize data """
        ret = self._df.multiply(self._norm, axis=axis, **kwargs)
        if inplace:
            self._df = ret
            return
        return ret

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
        self._pumped = value

    @property
    def probed(self):
        return self._probed

    @probed.setter
    def probed(self, value):
        if isinstance(self._df.get("bleach"), pd.core.series.Series):
            self.df.drop('bleach', axis=1, inplace=True)
        self.probed = value

    @property
    def pump(self):
        return self._pump

    @pump.setter
    def pump(self, value):
        self._pump = value   

    
class Scan(ScanBase):
    def __init__(self, df, base = None, norm = None, metadata=None):
        self._df = df
        self._base = base
        self._norm = norm
        self.metadata = metadata

    def model_base(self):
        raise NotImplemented
        left = self._df[:50].median(0)
        right = self._df[-50:].median(0)


    
class TimeScan(ScanBase, PumpProbe):
    def __init__(self, df, base=None, norm=None, pump=None, metadata=None,
                 pumped="spec_0", probed = "spec_1" ):
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
        return TimeScan(
            self._df.copy(),
            self._base,
            self._norm,
            self._pump,
            copy.deepcopy(self.metadata),
            self._pumped,
            self._probed)

    def sub_base(self, inplace=False, axis=0, level=1, **kwargs):
        """substract baseline from data"""
        ret = self._df.subtract(self._base, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
            return
        return ret

    def add_base(self, inplaxe=False, axis=0, level=1, **kwargs):
        """add baseline to data"""
        ret = self._df.add(self._base, axis=axis, level=level, **kwargs)
        if inplace:
            self._df = ret
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
        return self._df.index.levels[0]
    
    @property
    def groupby_pp_delay(self):
        return self._df.groupby(axis=0, level=0)
    
    @property
    def pumped(self):
        return self.med[self._pumped]

    @property
    def probed(self):
        return self.med[self._probed]

    @property
    def bleach(self):
        if isinstance(self._df.get("bleach"), type(None)):
            self._df["bleach"] = self.pumped - self.probed
        return self._df["bleach"]

    @bleach.setter
    def bleach(self, value):
        self._df["bleach"] = value


    
        
