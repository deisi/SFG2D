""" Module for clases, that infer a cetain content """
from .scan import ScanBase
from ..utils.detect_peaks import detect_peaks

class PumpVisSFG(ScanBase):
    
    def __init__(self, *args, spec=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec # The spectrum with the data.
        self._pp_delay_pos = 0 # With of the spectra is important

    @property
    def pp_delays(self):
        if 'pp_delay' in self._df.index.names:
            ret = self._df.index.levels[0]
        else:
            ret = 0
        return ret

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, str):
            self._spec = self.med[value]
        elif isinstance(value, type(None)):
            self._spec = self.med["spec_1"]
        else:
            return NotImplementedError

    @property
    def current(self):
        """ Selected representing spectra """
        if 'pp_delay' in self._df.index.names:
            current = self.spec.ix[self._pp_delay_pos]
        else:
            current = self.spec
        return current
        
        
    @property
    def freq(self):
        """ estimator for the frequency seted by the pump """
        r = self.current.rolling(5).median().idxmax()
        return r

    @property
    def width(self):
        """ estimator for the width of the pump """
        r = self.current.rolling(5).median().diff()
        return r.idxmin() - r.idxmax()
        

class IR(Scan):
    
    
