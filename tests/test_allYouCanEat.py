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
        del self.data

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
        # Must allow for small machine percision differences
        small_values = np.abs(wl - self.data.wavelength)
        assert np.any(small_values < 10**(-12))

    def test_data_wavenumber(self):
        wl = self.result_dict['wavenumber']
        # Must allow for small machine percision differences
        self.data.metadata["vis_wl"] = 810 # Set the right vis_wl
        small_values = np.abs(wl - self.data.wavenumber)
        assert np.any(small_values < 10**(-12))

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

class TestTs(unittest.TestCase):
    def setUp(self):
        self.data = AllYouCanEat('../data/09_ts_gold_w575_g1_e1s_ssp_pu1_pr1_vis1_gal1_chop1_purge1.dat')

    def tearDown(self):
        del self.data

    def test_wavenumber(self):
        self.data.wavenumber

    def test_wavelength(self):
        self.data.wavelength

    def test_pp_delays(self):
        pp_delays = np.array([
            -1900, -1700, -1500, -1300, -1100,  -900,  -700,  -500,  -400,
            -300,  -200,  -150,  -100,   -50,   -25,     0,    25,    50,
            100,   150,   200,   300,   400,   500,   700,   900,  1100,
            1300,  1500,  1700,  1900
        ], dtype=np.int)
        assert all(self.data.pp_delays == pp_delays)

    def test_shape_of_data(self):
        assert self.data.data.shape == (31, 3, 3, 1600)

    def test_frames(self):
        assert self.data.frames == 3

if __name__ == '__main__':
    sys.exit(unittest.main())
