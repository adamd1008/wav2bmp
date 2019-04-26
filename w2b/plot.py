# MIT License
#
# Copyright (c) 2019 Adam Dodd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from . import util


################################################################################
def draw_abs(name, fs, size, overlapDec, ab, inv=False, block=False):
    fig = plt.figure()
    fig.suptitle("abs [" + name + "]\nfs = " + str(fs) + \
            ", size = " + str(size) + ", ovl = " + str(overlapDec))

    if inv:
        ab2 = util.flip_norm(ab)
    else:
        ab2 = ab

    plt.imshow(ab2, cmap="gray", origin="lower")
    plt.show(block=block)


################################################################################
def draw_abs_db(name, fs, size, overlapDec, ab, inv=False, block=False):
    ab_db = util.mag2db_norm(ab)

    fig = plt.figure()
    fig.suptitle("dB(abs) [" + name + "]\nfs = " + str(fs) + \
            ", size = " + str(size) + ", ovl = " + str(overlapDec))

    if inv:
        ab_db2 = util.flip_norm(ab_db)
    else:
        ab_db2 = ab_db

    plt.imshow(ab_db2, cmap="gray", origin="lower")
    plt.show(block=block)


################################################################################
def draw_abs_db_log(name, fs, size, overlapDec, ab, inv=False, block=False):
    binFreqs, logFreqs = util.log_freq(fs, size)
    ab_db_log = util.lin2log(util.mag2db_norm(ab), binFreqs, logFreqs)

    fig = plt.figure()
    fig.suptitle("logY(dB(abs)) [" + name + "]\nfs = " + str(fs) + \
            ", size = " + str(size) + ", ovl = " + str(overlapDec))

    if inv:
        ab_db_log2 = util.flip_norm(ab_db_log)
    else:
        ab_db_log2 = ab_db_log

    plt.imshow(ab_db_log2, cmap="gray", origin="lower")
    plt.show(block=block)


################################################################################
def draw_ang(name, fs, size, overlapDec, an, block=False):
    an_norm = an / (2 * np.pi)

    assert np.amin(an_norm) >= 0.0
    assert np.amax(an_norm) <= 1.0

    fig = plt.figure()
    fig.suptitle("ang [" + name + "]\nfs = " + str(fs) + \
            ", size = " + str(size) + ", ovl = " + str(overlapDec))
    plt.imshow(an_norm, cmap="gray", origin="lower")
    plt.show(block=block)
