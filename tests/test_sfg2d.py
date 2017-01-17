#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sfg2d
----------------------------------

Tests for `sfg2d` module.
"""


import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

from sfg2d import sfg2d
from sfg2d import cli

import numpy as np


class TestAllYouCanEat(unittest.TestCase):

    def setUp(self):
        from sfg2d.io.allYouCanEat import AllYouCanEat
        self.data = AllYouCanEat('../sfg2d/data/00_sp_quarz_w650_gcm_e20s_pr3000.dat')

    def tearDown(self):
        pass

    def test_pp_delays_is_numpy_array(self):
        assert isinstance(self.data, type(np.zeros(1)))



if __name__ == '__main__':
    sys.exit(unittest.main())
