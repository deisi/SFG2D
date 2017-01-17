import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

import numpy as np
import sfg2d.utils.metadata as metadata


class TestUtilsMetadata(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_metadata_from_filename(self):
        from datetime import timedelta
        test_str = [
            '00_sp_quartz_w650_g139_e100ms_ppp_pump1.dat',
            '03_ts_gold_wl345_gcm_e10m_ssp_probe1.dat',
            'kjshdfksdkue_jsjsjs912921342734_jaajqw'
        ]
        md = metadata.get_metadata_from_filename(test_str[0])
        assert md['sp_type'] == 'sp'
        assert md['exposure_time'] == timedelta(0, 0, 100000)
        assert md['gain'] == 139
        assert md['material'] == 'quartz'
        assert md['polarisation'] == 'ppp'
        assert md['central_wl'] == 650

        md = metadata.get_metadata_from_filename(test_str[1])
        assert md['central_wl'] == 345
        assert md['gain'] == -1
        assert md['material'] == 'gold'
        assert md['polarisation'] == 'ssp'
        assert md['sp_type'] == 'ts'
        assert md['exposure_time'] == timedelta(0, 600)

        md = metadata.get_metadata_from_filename(test_str[2])
        assert md['uri']



if __name__ == '__main__':
    sys.exit(unittest.main())
