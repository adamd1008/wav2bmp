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

import imageio as iio
import numpy as np

from . import colourmap as cm
from . import util


################################################################################
def write_abs(name, fs, size, overlapDec, ab, inv=False):
    baseName = name + "_fs" + str(fs) + "_s" + str(size) + \
            "_o" + str(overlapDec) + "_ab"
    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    if inv:
        ab2 = util.flip_norm(ab)
    else:
        ab2 = ab

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab2)


################################################################################
def write_abs_ocl(name, fs, size, bins, startFreq, endFreq,
        overlapDec, ab, inv=False):
    baseName = name + "_fs" + str(fs) + "_s" + str(size) + "_b" + str(bins) + \
            "_sf" + str(startFreq) + "_ef" + str(endFreq) + \
            "_o" + str(overlapDec) + "_ab"
    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    if inv:
        ab2 = util.flip_norm(ab)
    else:
        ab2 = ab

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab2)


################################################################################
def write_abs_db(name, fs, size, overlapDec, ab, inv=False):
    ab_db = util.mag2db_norm(ab)
    baseName = name + "_fs" + str(fs) + "_s" + str(size) + \
            "_o" + str(overlapDec) + "_ab_db"
    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    if inv:
        ab_db2 = util.flip_norm(ab_db)
    else:
        ab_db2 = ab_db

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab_db2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab_db2)


################################################################################
def write_abs_db_ocl(name, fs, size, bins, startFreq, endFreq,
        overlapDec, ab, inv=False):
    ab_db = util.mag2db_norm(ab)
    baseName = name + "_fs" + str(fs) + "_s" + str(size) + "_b" + str(bins) + \
            "_sf" + str(startFreq) + "_ef" + str(endFreq) + \
            "_o" + str(overlapDec) + "_ab_dB"
    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    if inv:
        ab_db2 = util.flip_norm(ab_db)
    else:
        ab_db2 = ab_db

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab_db2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab_db2)


################################################################################
def write_abs_db_log(name, fs, size, overlapDec, ab, inv=False):
    binFreqs, logFreqs = util.log_freq(fs, size)
    ab_db_log = util.lin2log(util.mag2db_norm(ab), binFreqs, logFreqs)
    baseName = name + "_fs" + str(fs) + "_s" + str(size) + \
            "_o" + str(overlapDec) + "_ab_db_log"
    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    if inv:
        ab_db_log2 = util.flip_norm(ab_db_log)
    else:
        ab_db_log2 = ab_db_log

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab_db_log2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab_db_log2)


################################################################################
def write_ang(name, fs, size, overlapDec, ab, an,
        bins=None, startFreq=None, endFreq=None,
        colourMap=cm.colour_maps["thermal1"], normAbs=True):
    """Write the FFT phase information scaled by amplitude"""

    if normAbs:
        abNorm = util.norm(ab)
    else:
        abNorm = ab

    anNorm = an / (2 * np.pi)

    if (bins == None) or (startFreq == None) or (endFreq == None):
        binFreqs, logFreqs = util.log_freq(fs, size)

        if bins == None:
            bins = len(binFreqs)

        if startFreq == None:
            startFreq = binFreqs[0]

        if endFreq == None:
            endFreq = binFreqs[-1]

    baseName = name + "_fs" + str(fs) + "_s" + str(size) + "_b" + str(bins) + \
            "_sf" + str(startFreq) + "_ef" + str(endFreq) + \
            "_o" + str(overlapDec) + "_an"

    if normAbs:
        baseName += "_norm"

    imgName = baseName + ".bmp"
    rawName = baseName + ".npy"

    assert np.amin(abNorm) >= 0.0
    assert np.amax(abNorm) <= 1.0
    assert np.amin(anNorm) >= 0.0
    assert np.amax(anNorm) <= 1.0

    img = np.ndarray((abNorm.shape[0], abNorm.shape[1], 3), dtype="float32")

    img[:, :, 0] = abNorm[:, :] * \
            np.interp(anNorm[:, :], colourMap["r"]["x"], colourMap["r"]["y"])
    img[:, :, 1] = abNorm[:, :] * \
            np.interp(anNorm[:, :], colourMap["g"]["x"], colourMap["g"]["y"])
    img[:, :, 2] = abNorm[:, :] * \
            np.interp(anNorm[:, :], colourMap["b"]["x"], colourMap["b"]["y"])

    assert np.amin(img) >= 0.0
    assert np.amax(img) <= 1.0

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(img))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, img)
