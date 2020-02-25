#!/usr/bin/python3

import re
import unittest
import numpy as np

import w2b.fft as fft


################################################################################
class TestFftStatsParam(unittest.TestCase):
    """Parameters: ((n, size, overlapDec), start, step, iters)"""
    @classmethod
    def setUpClass(cls):
        cls.param_list = [
                (( 39,   8, 0.0   ), (    0,   8,    5)),
                (( 39,   8, 0.5   ), (   -4,   4,   11)),
                (( 39,   8, 0.75  ), (   -6,   2,   23)),
                ((  5,   4, 0.75  ), (   -3,   1,    8)),
                ((  9,   8, 0.0   ), (    0,   8,    2)),
                ((  9,   8, 0.5   ), (   -4,   4,    4)),
                ((  9,   8, 0.75  ), (   -6,   2,    8)),
                ((  9,   8, 0.875 ), (   -7,   1,   16))
        ]

    def test_fft_stats(self):
        for params, expected in self.param_list:
            with self.subTest(
                    msg="n={}, size={}, overlapDec={}".format(*params)):

                actual = fft.get_fft_stats(*params)
                self.assertEqual(expected, actual)


################################################################################
class TestFftStatsErrors(unittest.TestCase):
    """Parameters: ((n, size, overlapDec), regex)"""
    @classmethod
    def setUpClass(cls):
        cls.param_list = [
                ((1,  2, 0.0 ), "`n' cannot be less than size"),
                ((2,  2, 1.0 ), "Overlap must be LT 1.0, GE 0.0"),
                ((2,  2, 0.87), "Overlap must be 1 / (2^n)"),
                ((1,  1, 0.0 ), "Size must be GE 2"),
                ((1, -1, 0.0 ), "Size must be GE 2"),
                ((4,  3, 0.0 ), "Size must be 2^n")
        ]

    def test_fft_stats_errors(self):
        for params, errStr in self.param_list:
            with self.subTest(
                    msg="n={}, size={}, overlapDec={}".format(*params)):

                self.assertRaisesRegex(
                        ValueError, re.escape(errStr),
                        fft.get_fft_stats, *params)


################################################################################
if __name__ == "__main__":
    unittest.main()
