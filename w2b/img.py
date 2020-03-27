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

import imageio as iio
import numpy as np

from . import colourmap as cm
from . import util


################################################################################
def write_abs(
        name, fs, size, overlapDec, ab,
        bins=None, startFreq=None, endFreq=None):
    """Write FT amplitude image and data to disk."""

    imgName = util.gen_filename(
            name, fs, size, overlapDec, "ab", "bmp", False,
            bins, startFreq, endFreq)
    rawName = util.gen_filename(
            name, fs, size, overlapDec, "ab", "npy", False,
            bins, startFreq, endFreq)

    ab2 = util.convert_to_img_type(ab)

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab2)


################################################################################
def write_abs_db(
        name, fs, size, overlapDec, ab,
        bins=None, startFreq=None, endFreq=None):
    """Write FT decibel amplitude image and data to disk."""

    imgName = util.gen_filename(
            name, fs, size, overlapDec, "ab-dB", "bmp", False,
            bins, startFreq, endFreq)
    rawName = util.gen_filename(
            name, fs, size, overlapDec, "ab-dB", "npy", False,
            bins, startFreq, endFreq)

    ab_db = util.mag2db_norm(ab)
    ab_db2 = util.convert_to_img_type(ab_db)

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab_db2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab_db2)


################################################################################
def write_abs_db_log(name, fs, size, overlapDec, ab):
    """Write FT decibel amplitude image and data to disk with logarithmic
    frequency."""

    imgName = util.gen_filename(
            name, fs, size, overlapDec, "ab-dB-log", "bmp", False,
            bins, startFreq, endFreq)
    rawName = util.gen_filename(
            name, fs, size, overlapDec, "ab-dB-log", "npy", False,
            bins, startFreq, endFreq)

    binFreqs, logFreqs = util.log_freq(fs, size)
    ab_db_log = util.lin2log(util.mag2db_norm(ab), binFreqs, logFreqs)
    ab_db_log2 = util.convert_to_img_type(ab_db_log)

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(ab_db_log2))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, ab_db_log2)


################################################################################
def write_ang(name, fs, size, overlapDec, ab, an,
        bins=None, startFreq=None, endFreq=None,
        colourMap=cm.colour_maps["thermal1"], normAbs=True):
    """Write the FT phase information with colourmap applied."""

    imgName = util.gen_filename(
            name, fs, size, overlapDec, "an", "bmp", False,
            bins, startFreq, endFreq)
    rawName = util.gen_filename(
            name, fs, size, overlapDec, "an", "npy", False,
            bins, startFreq, endFreq)

    if normAbs:
        ab2 = util.norm(ab)
    else:
        ab2 = ab

    img = util.convert_to_img_type(util.apply_colourmap(ab2, an, colourMap))

    print("Writing image file \"" + imgName + "\"")
    iio.imwrite(imgName, np.flipud(img))

    print("Writing raw file \"" + rawName + "\"")
    np.save(rawName, img)
