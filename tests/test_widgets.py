#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_widgets
----------------------------------

Tests for `sfg2d` module.
"""


import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

from ipykernel.comm import Comm
from ipywidgets import Widget
import numpy as np

from sfg2d.widgets import SpecAndBase, SpecAndSummed, Normalized

###################################################
# Stuff needed to run widgets within a terminal
###################################################

class DummyComm(Comm):
    comm_id = 'a-b-c-d'

    def open(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

_widget_attrs = {}
undefined = object()

def setup_test_comm():
    _widget_attrs['_comm_default'] = getattr(Widget, '_comm_default', undefined)
    Widget._comm_default = lambda self: DummyComm()
    _widget_attrs['_ipython_display_'] = Widget._ipython_display_
    def raise_not_implemented(*args, **kwargs):
        raise NotImplementedError()
    Widget._ipython_display_ = raise_not_implemented

def teardown_test_comm():
    for attr, value in _widget_attrs.items():
        if value is undefined:
            delattr(Widget, attr)
        else:
            setattr(Widget, attr, value)

###################################################
# The actual tests
###################################################

class TestSpecAndBase(unittest.TestCase):
    def setUp(self):
        setup_test_comm()
        self.widget = SpecAndBase()
        self.widget()
        self.widget.wTextFolder.value = '../data'
        self.widget._on_folder_submit(None)
        self.widget.wSelectFile.value = '09_ts_gold_w575_g1_e1s_ssp_pu1_pr1_vis1_gal1_chop1_purge1.dat'
        self.widget.wSelectBaseFile.value = '09_ts_gold_w575_g1_e1s_ssp_pu1_pr1_vis1_gal1_chop1_purge1.dat'

    def tearDown(self):
        teardown_test_comm()

    def test_pp_s_options(self):
        res = [-1900.0, -1700.0, -1500.0, -1300.0, -1100.0, -900.0, -700.0,
               -500.0, -400.0, -300.0, -200.0, -150.0, -100.0, -50.0, -25.0,
               0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0,
               900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0]

        assert np.all(self.widget.wSliderPPDelay.options == res)

    def test_frame_slider_max(self):
        assert self.widget.wSliderFrame.max == 2

    def test_x_pixel_range_max(self):
        assert self.widget.wIntRangeSliderPixelX.max == 1600

    def test_y_pixel_range_max(self):
        assert self.widget.wIntRangeSliderPixelY.max == 3

    def test_frame_baseline_max(self):
        assert self.widget.wSliderBaselineFrame.max == 2

    def test_pp_baseline_slider_max(self):
        assert self.widget.wSliderBaselinePPDelay.max == 30

    def test_spec_base_slider(self):
        assert self.widget.wRangeSliderBaselineSpec.max == 3

    def test_sub_baseline_methods(self):
        # Activate the baseline substraction and
        # Change some properties of the baseline
        self.widget.wToggleSubBaseline.value = True
        self.widget.wRangeSliderBaselineSpec.value = (0, 3)
        self.widget.wRangeSliderBaselineSpec.value = (0, 1)
        self.widget.wSliderBaselineFrame.value = 2
        self.widget.wCheckBaselineFrameMedian.value = True
        self.widget.wToggleSubBaseline.value = False

    def test_data_selection_methods(self):
        self.widget.wSliderPPDelay.value = 25.0
        self.widget.wSliderFrame.value = 2
        self.widget.wCheckFrameMedian.value = True
        self.widget.wIntSliderSmooth.value = 5
        assert self.widget.y.shape == (1600, 1)
        self.widget.wIntRangeSliderPixelY.value = (0, 2)
        assert self.widget.y.shape == (1600, 2)
        self.widget.wIntRangeSliderPixelX.value = (234, 1000)

    def test_calibrations(self):
        self.widget.wDropdownCalib.value = "nm"
        self.widget.wDropdownCalib.value = "wavenumber"
        self.widget.wTextVisWl.value = 800

class TestSpecAndSummed(unittest.TestCase):
    def setUp(self):
        setup_test_comm()
        self.widget = SpecAndSummed()
        self.widget()
        self.widget.wTextFolder.value = '../data'
        self.widget._on_folder_submit(None)
        self.widget.wSelectFile.value = '09_ts_gold_w575_g1_e1s_ssp_pu1_pr1_vis1_gal1_chop1_purge1.dat'

    def tearDown(self):
        teardown_test_comm()

    def test_sum_over_toggle(self):
        self.widget.wDropSumAxis.value = 'pp_delays'
        assert all(self.widget.x_sum == self.widget.data.pp_delays)
        assert self.widget.y_sum.size == self.widget.x_sum.size

        self.widget.wDropSumAxis.value = 'frames'
        assert self.widget.x_sum.size == self.widget.data.data.shape[-3]
        assert self.widget.y_sum.size == self.widget.x_sum.size


class TestNormalized(unittest.TestCase):

    def setUp(self):
        setup_test_comm()
        self.widget = Normalized()

    def tearDown(self):
        teardown_test_comm()

    def test_if_it_loads(self):
        assert self.widget

if __name__ == '__main__':
    sys.exit(unittest.main())
