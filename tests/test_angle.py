#!/usr/bin/python3

import unittest
import numpy as np

import w2b.util as util


################################################################################
class TestAngle(unittest.TestCase):
    def check_angle(self, a, out):
        res = util.angle(a)
        self.assertTrue(np.allclose(res, out))

    def test_angle(self):
        self.check_angle(0.0+0j, 0.0)
        self.check_angle(np.array([0.0+0j, 0.0+0j]), [0.0, 0.0])
        self.check_angle(np.array([[0.0+0j, 0.0+0j], [0.0+0j, 0.0+0j]]),
                [[0.0, 0.0], [0.0, 0.0]])
        self.check_angle(
                np.array(
                    [[0.0+0j, 0.0+0j, 0.0+0j],
                     [0.0+0j, 0.0+0j, 0.0+0j]]),
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # TODO


################################################################################
if __name__ == "__main__":
    unittest.main()
