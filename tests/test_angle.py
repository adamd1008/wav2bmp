#!/usr/bin/python3

import re
import unittest
import numpy as np

import w2b.util as util


################################################################################
class TestAngleParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_list = [
                ( 0.0, 0.0),
                ( 0.0+0.0j, 0.0),
                (np.array([ 0.0+0.0j,  0.0+0.0j]),
                    [0.0, 0.0]),
                (np.array([[ 0.0+0.0j,  0.0+0.0j], [ 0.0+0.0j,  0.0+0.0j]]),
                    [[0.0, 0.0], [0.0, 0.0]]),
                (np.array([[ 0.0+0.0j,  0.0+0.0j,  0.0+0.0j],
                        [ 0.0+0.0j,  0.0+0.0j,  0.0+0.0j]]),
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),

                ( 1.0     , 0.0  ),
                ( 1.0+0.0j, 0.0  ),
                ( 1.0+1.0j, 0.125),
                ( 0.0+1.0j, 0.25 ),
                (-1.0+1.0j, 0.375),
                (-1.0+0.0j, 0.5  ),
                (-1.0-1.0j, 0.625),
                ( 0.0-1.0j, 0.75 ),
                ( 1.0-1.0j, 0.875),

                (np.array([ 1.0+0.0j]), [0.0  ]),
                (np.array([ 1.0+1.0j]), [0.125]),
                (np.array([ 0.0+1.0j]), [0.25 ]),
                (np.array([-1.0+1.0j]), [0.375]),
                (np.array([-1.0+0.0j]), [0.5  ]),
                (np.array([-1.0-1.0j]), [0.625]),
                (np.array([ 0.0-1.0j]), [0.75 ]),
                (np.array([ 1.0-1.0j]), [0.875]),

                (np.array(
                    [ 1.0+0.0j,  1.0+1.0j,  0.0+1.0j, -1.0+1.0j,
                     -1.0+0.0j, -1.0-1.0j,  0.0-1.0j,  1.0-1.0j]),
                    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]),
                (np.array(
                    [[ 1.0+0.0j,  1.0+1.0j,  0.0+1.0j, -1.0+1.0j],
                     [-1.0+0.0j, -1.0-1.0j,  0.0-1.0j,  1.0-1.0j]]),
                    [[0.0, 0.125, 0.25, 0.375], [0.5, 0.625, 0.75, 0.875]])
        ]

    def check_angle(self, a, out):
        temp = util.angle(a)
        self.assertTrue(np.allclose(temp, out))

    def test_angle(self):
        for a, out in self.param_list:
            with self.subTest(msg="{} | {}".format(a, out)):
                self.check_angle(a, out)


################################################################################
class TestAngleErrors(unittest.TestCase):
    def test_angle_error_dims(self):
        a = np.ndarray((2, 2, 2), dtype=complex)
        self.assertRaisesRegex(
                ValueError, re.escape("Expected 1 <= ndim <= 2"), util.angle, a)

    def test_angle_error_assert_type(self):
        a = np.ndarray((2, 2), dtype="float32")
        self.assertRaises(AssertionError, util.angle, a)


################################################################################
if __name__ == "__main__":
    unittest.main()
