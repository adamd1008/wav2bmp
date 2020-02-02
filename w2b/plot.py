# MIT License
#
# Copyright (c) 2020 Adam Dodd
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

import matplotlib.pyplot as plt
import numpy as np

from . import colourmap as cm
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
def draw_ang(name, fs, size, overlapDec, ab, an,
        bins=None, startFreq=None, endFreq=None,
        colourMap=cm.colour_maps["thermal1"], normAbs=True, block=False):
    """Draw the FFT phase information scaled by amplitude"""

    if normAbs:
        abNorm = util.norm(ab)
    else:
        abNorm = ab

    if (bins == None) or (startFreq == None) or (endFreq == None):
        binFreqs, logFreqs = util.log_freq(fs, size)

        if bins == None:
            bins = len(binFreqs)

        if startFreq == None:
            startFreq = binFreqs[0]

        if endFreq == None:
            endFreq = binFreqs[-1]

    assert np.amin(abNorm) >= 0.0
    assert np.amax(abNorm) <= 1.0
    assert np.amin(an) >= 0.0
    assert np.amax(an) <  1.0

    img = np.ndarray((abNorm.shape[0], abNorm.shape[1], 3), dtype="float32")

    img[:, :, 0] = abNorm[:, :] * \
            np.interp(an[:, :], colourMap["r"]["x"], colourMap["r"]["y"])
    img[:, :, 1] = abNorm[:, :] * \
            np.interp(an[:, :], colourMap["g"]["x"], colourMap["g"]["y"])
    img[:, :, 2] = abNorm[:, :] * \
            np.interp(an[:, :], colourMap["b"]["x"], colourMap["b"]["y"])

    assert np.amin(img) >= 0.0
    assert np.amax(img) <= 1.0

    graphName = "ang [" + name + "]\nfs = " + str(fs) + \
            ", size = " + str(size) + ", bins = " + str(bins) + \
            ", startFreq = " + str(startFreq) + \
            ", endFreq = " + str(endFreq) + ", ovl = " + str(overlapDec)

    if normAbs:
        graphName += " (ab norm)"

    fig = plt.figure()
    fig.suptitle(graphName)
    plt.imshow(img, origin="lower")
    plt.show(block=block)
