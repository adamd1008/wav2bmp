#!/usr/bin/python3

import re
import unittest
import numpy as np

import w2b.util as util


# TODO:
# (1) Tests to convert fields to filename.
# (2) Tests to convert filename to fields and back again.
# (3) Tests to test errors raised.


################################################################################
class TestFileNameToNameParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_list = [
                ("test_wav.wav___fs44100_s4096_b2049_" +
                 "sf0.0_ef22050.0_o0.875_ab.npy"),
                ("test_wav.wav___fs44100.0_s4096_b2049_" +
                 "sf0.0_ef22050.0_o0.875_an_norm.bmp")
        ]

    def test_parse_filename_to_name(self):
        for fileName in self.param_list:
            with self.subTest(msg=fileName):
                actual = util.gen_filename_w_dict(
                        util.parse_filename(fileName))
                self.assertEqual(fileName, actual)


################################################################################
class TestFileNameToFieldsParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Params: (fileName, {... expected fields ...})
        cls.param_list = [
                ("test_wav.wav___fs44100_s4096_b2049_" +
                 "sf0.0_ef22050.0_o0.875_ab.npy",
                {
                    'fileName': "test_wav.wav",
                    'fileExt': "npy",
                    'sampleRate': 44100.0,
                    'size': 4096,
                    'bins': 2049,
                    'startFreq': 0.0,
                    'endFreq': 22050.0,
                    'overlapDec': 0.875,
                    'fileType': "ab",
                    'isNorm': False
                }),
                ("test_wav.wav___fs44100.0_s4096_b2049_" +
                 "sf0.0_ef22050.0_o0.875_an_norm.bmp",
                {
                    'fileName': "test_wav.wav",
                    'fileExt': "bmp",
                    'sampleRate': 44100.0,
                    'size': 4096,
                    'bins': 2049,
                    'startFreq': 0.0,
                    'endFreq': 22050.0,
                    'overlapDec': 0.875,
                    'fileType': "an",
                    'isNorm': True
                })
        ]

    def test_parse_filename_to_fields(self):
        for fileName, expected in self.param_list:
            with self.subTest(msg=fileName):
                actual = util.parse_filename(fileName)
                self.assertEqual(expected, actual)


################################################################################
if __name__ == "__main__":
    unittest.main()
