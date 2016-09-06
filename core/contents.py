""" Module for clases, that infer a cetain content """
from .scan import ScanBase, Scan
from ..utils.detect_peaks import detect_peaks

class ContenClass():
    _spec = None
    _current = None
    
    @property
    def spec(self):
        """A selected Spectrum """
        return self._spec

    @spec.setter
    def spec(self, value):
        self._spec = value

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        self._current = value

    @property
    def freq(self):
        raise NotImplementedError

    @property
    def width(self):
        raise NotImplementedError


class PumpVisSFG(ScanBase, ContenClass):
    
    def __init__(self, *args, spec=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec # The spectrum with the data.
        self.ppdelay_current = self.pp_delays[0]

    @property
    def pp_delays(self):
        if 'pp_delay' in self._df.index.names:
            ret = self._df.index.levels[0]
        else:
            ret = [0]
        return ret

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        # Force self.med to be recalculated
        self._med = None
        if isinstance(value, str):
            if value is 'All':
                self._spec = self.med
            else:
                self._spec = self.med[value]
        elif isinstance(value, type(None)):
            self._spec = self.med["spec_1"]
        else:
            return NotImplementedError

    @property
    def current(self):
        """ Selected representing spectra """
        # guarantee it has pp_delays in it
        # because we can also load spectra and
        # scans here
        if 'pp_delay' in self._df.index.names:
            current = self.spec.ix[self.ppdelay_current]
        else:
            current = self.spec
        return current
        
    @property
    def freq(self):
        """ estimator for the frequency seted by the pump """
        r = self.current.rolling(10).median().idxmax()
        return r

    @property
    def width(self):
        """ estimator for the width of the pump """
        r = self.current.rolling(10).median().diff()
        return abs(r.idxmin() - r.idxmax())
        

class IR(Scan, ContenClass):
    def __init__(self, *args, spec='spec_1', **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec # Identifier of spectrum with the data.
        self._pp_delay_pos = 0 

        
    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, str):
            # Reset med to make sure its up to date
            self._med = None
            if value is "All":
                self._spec = self.med
            else:
                self._spec = self.med[value]
        #elif isinstance(value, type(None)):
        #    self._spec = self.med["spec_1"]
        else:
            return NotImplementedError

    @property
    def current(self):
        return self.spec

    @property
    def freq(self):
        """ estimator for the frequency seted by the pump """
        r = self.current.rolling(10).median().idxmax()
        return r

    @property
    def width(self):
        """ estimator for the width of the pump """
        r = self.current.rolling(20).median().diff()
        return abs(r.idxmin() - r.idxmax())
        

    
