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

import numpy as np
from numpy.fft import rfftfreq


################################################################################
def norm(ar):
    if ar.dtype != "float32":
        ret = ar.astype("float32", casting="safe")
        return ret / np.iinfo(ar.dtype).max
    else:
        arMin = np.abs(np.amin(ar))
        arMax = np.abs(np.amax(ar))

        if arMin > arMax:
            return ar / arMin
        else:
            return ar / arMax


################################################################################
def mag2db(ar):
    db = 20.0 * np.ma.log10(ar, dtype="float32")
    return db.filled(0.0)


################################################################################
def mag2db_norm(ar):
    db = 20.0 * np.ma.log10(ar, dtype="float32")
    dbMin = 20.0 * np.log10(1.0 / np.power(2.0, 32.0))
    dbActualMin = np.amin(db)

    if dbActualMin < dbMin:
        #print("Warning: dbActualMin < dbMin")
        dbMin = dbActualMin

    ret = (db / -dbMin) + 1.0

    return ret.filled(0.0)


################################################################################
def flip_norm(ar):
    return 1.0 - ar


################################################################################
def log_freq(fs, size):
    nf = fs / 2.0
    fftSize = int(size / 2) + 1

    # What x value on the 2^x graph results in the Nyquist frequency (nf)?
    x = np.log2(nf)

    # Use the number of bins, and evenly divide them by the x-range
    # Subtract 1 from `fftSize` so that the `logFreqs` range is 1 to nf, incl.
    xPerBin = x / float(fftSize - 1)
    binFreqs = rfftfreq(size, d=(1.0 / fs))

    assert binFreqs.ndim == 1
    assert fftSize == binFreqs.shape[0]

    # Generate log freq-axis
    logFreqs = np.ndarray(fftSize, dtype="float32")

    for i in range(0, fftSize):
        logFreqs[i] = np.power(2.0, xPerBin * i)

    return binFreqs, logFreqs


################################################################################
def lin2log(ab, binFreqs, logFreqs):
    l = ab.shape[0]
    assert binFreqs.shape[0] == l
    assert logFreqs.shape[0] == l

    if len(ab.shape) == 1:
        ret = np.ndarray(l, dtype="float32")
        ret[:] = np.interp(logFreqs, binFreqs, ab)
    elif len(ab.shape) == 2:
        ret = np.ndarray(ab.shape, dtype="float32")

        for j in range(0, ab.shape[1]):
            ret[:, j] = np.interp(logFreqs, binFreqs, ab[:, j])
    else:
        raise ValueError("Unknown shape")

    return ret


################################################################################
def angle(x):
    """A function that takes a complex `ndarray` and returns the FFT-friendly
    angles.

    The NumPy `angle()` function returns negative angles when there is a -ve
    imaginary component; I want 0 to 2pi when rotating all the way round
    anti-clockwise from 1+0j.

    Normalises return values (0 <= ret < 1).
    """

    pi2 = (2.0 * np.pi)
    ang = np.angle(x)

    if type(ang) == np.ndarray:
        assert x.dtype == complex
        assert ang.shape == x.shape

        if ang.ndim == 1:
            for i in range(0, ang.shape[0]):
                if ang[i] < 0.0:
                    ang[i] = ang[i] + pi2
        elif ang.ndim == 2:
            for i in range(0, ang.shape[0]):
                for j in range(0, ang.shape[1]):
                    if ang[i, j] < 0.0:
                        ang[i, j] = ang[i, j] + pi2
        else:
            raise ValueError("Expected 1 <= ndim <= 2")
    else:
        if ang < 0.0:
            ang = ang + pi2

    return ang / pi2
