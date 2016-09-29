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



class TestSfg2d(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_000_something(self):
        pass

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'sfg2d.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output


if __name__ == '__main__':
    sys.exit(unittest.main())
