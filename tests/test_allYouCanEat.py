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

import numpy as np
from datetime import timedelta
from sfg2d.io.allYouCanEat import AllYouCanEat

class TestQuartz(unittest.TestCase):

    def setUp(self):
        self.data = AllYouCanEat(
            '../sfg2d/data/00_sp_quarz_w650_gcm_e20s_pr3000.dat')
        self.result_dict = {
            'shape_of_data' : (1, 1, 3, 1600),
            'metadata' : {
                'central_wl' : 650,
                'material' : 'quarz',
                'sp_type' : 'sp',
                'gain' : -1,
                'exposure_time' : timedelta(0, 20),
            },
            'some_row' : ([0, 0, 1, slice(None, None)],
                          np.load('../data/00_quarts_row_1.npy')),
            'type' : 'sp',
            'pixel' : np.arange(1600),
            'times' : [timedelta(0)],
            'frames' : 1,
            'pp_delays' : np.array([0]),
            'wavelength' : np.load('../data/00_quartz_wavelength.npy'),
            'wavenumber' : np.load('../data/00_quartz_wavenumber.npy'),
        }

    def tearDown(self):
        pass

    def test_pp_delays_is_numpy_array(self):
        assert isinstance(self.data.pp_delays, type(np.zeros(1)))

    def test_data_is_numpy_array(self):
        assert isinstance(self.data.data, type(np.zeros(1)))

    def test_shape_of_data(self):
        assert self.data.data.shape == self.result_dict['shape_of_data']

    def test_metadata(self):
        md = self.data.metadata
        for key in self.result_dict['metadata']:
            assert self.data.metadata[key] == self.result_dict['metadata'][key]

    def test_some_row(self):
        ind, data = self.result_dict['some_row']
        assert np.all(self.data.data[ind] == data)

    def test_type(self):
        assert self.data._type is self.result_dict['type']

    def test_data_pixel(self):
        assert all(self.data.pixel == self.result_dict['pixel'])

    def test_data_times(self):
        assert self.data.times == self.result_dict['times']

    def test_data_frames(self):
        assert self.data.frames == self.result_dict['frames']

    def test_data_ppdelays(self):
        assert self.data.pp_delays == self.result_dict['pp_delays']

    def test_data_wavelength(self):
        wl = self.result_dict['wavelength']
        assert all(self.data.wavelength == wl)

    def test_data_wavenumber(self):
        wl = self.result_dict['wavenumber']
        assert all(self.data.wavenumber == wl)

class TestSPE(TestQuartz):

    def setUp(self):
        self.data = AllYouCanEat('../data/08_h2o_gcm_e10m_ssp_purged1_pr6150nm_background.spe')
        self.result_dict = {
            'shape_of_data' : (1, 60, 1, 1600),
            'metadata' : {
                'exposure_time' : timedelta(0, 599, 945984),
                'material' : 'h2o',
                'polarisation' : 'ssp',
                'sp_type' : 'spe',
                'tempSet' : -75.0,
            },
            'some_row' : ([0, 23, 0, slice(None, None)],
                          np.load('../data/08_frame23.npy')),
            'type' : 'spe',
            'pixel' : np.arange(1600),
            'times' : np.load('../data/08_times.npy').tolist(),
            'frames' : 60,
            'pp_delays' : np.array([0]),
            'wavelength' : np.load('../data/08_wavelength.npy'),
            'wavenumber' : np.load('../data/08_wavenumber.npy'),
        }

if __name__ == '__main__':
    sys.exit(unittest.main())
