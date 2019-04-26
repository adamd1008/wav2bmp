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

import numpy as np
from numpy.fft import rfft, irfft

from . import util


################################################################################
def get_fft_stats(n, size, overlapDec):
    """Calculate the stats needed to iteratively compute the FFT over a set of
    wave samples with a given length, size and overlap.
    """

    if n < size:
        raise ValueError("`n' cannot be less than size")

    if (overlapDec >= 1.0) or (overlapDec < 0.0):
        raise ValueError("Overlap must be LT 1.0, GE 0.0")
    elif overlapDec != 0.0:
        overlapLog2 = np.log2(1.0 / (1.0 - overlapDec))

        if int(overlapLog2) != overlapLog2:
            raise ValueError("Overlap must be 1 / (2^n)")

    if size < 2:
        raise ValueError("Size must be GE 2")

    sizeLog2 = np.log2(size)

    if np.abs(sizeLog2) != sizeLog2:
        raise ValueError("Size must be 2^n")

    sizeOverlap = size * overlapDec

    if int(sizeOverlap) != sizeOverlap:
        raise ValueError("Size is not wholly divisible by overlap")

    # So, start the spectrogram with size * overlapDec's worth of samples not
    # in the FFT (4 zeros 50%, 6 zeroes 75%, 7 zeroes 87.5%, etc.)
    overlap = int(size * overlapDec)
    step = size - overlap

    if int(step) != step:
        raise ValueError("`overlapDec' must produce whole step")

    nDivStep = int(n / step)
    nModStep = int(n % step)
    sizeDivStep = int(size / step)

    # Left iterations is going to be one less than the number of times that
    # step goes into size
    leftPadIters = sizeDivStep - 1
    start = -(leftPadIters * step)

    # Right iterations is a little harder; we need to check how many samples are
    # left over when dividing all samples by step (not size, right?)

    iters = nDivStep + leftPadIters

    if nModStep > 0:
        iters += 1

    # Debug prints
    #print("n =", n)
    #print("size =", size)
    #print("overlapDec =", overlapDec)
    #print("overlap =", overlap)
    #print("step =", step)
    #print("leftPadIters =", leftPadIters)
    #print("start =", start)
    #print("iters =", iters)
    #print()

    #iterString = ""
    #iterStringFirst = True
    #count = 0

    #for i in range(start, n, step):
    #    if iterStringFirst:
    #        iterStringFirst = False
    #    else:
    #        iterString += ", "

    #    iterString += str(i) + " to " + str(i + size)
    #    count += 1

    #print(str(count) + " iters: ", iterString)
    #print()

    return start, n, step, iters


################################################################################
def wav2bmp(fs, wav, size=1024, overlapDec=0.0, window=np.hanning):
    """Transform wave samples into a spectrogram image.
    """

    if wav.ndim != 1:
        raise ValueError("Expected 1-dim array")
    l = wav.shape[0]

    fftLen = int(size / 2) + 1

    if callable(window):
        wnd = window(size)
    elif type(window) == np.ndarray:
        if window.ndim != 1:
            raise ValueError("Expected `window' to be a 1-dim array")
        elif window.shape[0] != size:
            raise ValueError("Expected `window' to be `size/2+1'")
        else:
            wnd = window
    elif window == None:
        wnd = None
    else:
        raise ValueError("Expected `window' to be a function or NumPy array")

    start, n, step, iters = get_fft_stats(l, size, overlapDec)
    c = 0
    ab = np.zeros((fftLen, iters), dtype="float32")
    an = np.zeros((fftLen, iters), dtype="float32")
    x = np.zeros((fftLen, iters), dtype=complex)
    buf = np.ndarray(size, dtype="float32")

    for i in range(start, n, step):
        # Do stuff

        if i < 0:
            bufStart = size - (size + i)
            bufEnd = size
            wavEnd = i + size

            #print("[<] i =", i, "\tbuf: 0 to", bufStart, "to", bufEnd, \
            #        "\twav: 0 to", wavEnd, "|", wavEnd)

            buf[0:bufStart] = 0.0
            buf[bufStart:bufEnd] = wav[0:wavEnd]
        elif (i + size) >= n:
            bufEnd = n - i
            wavStart = i

            #print("[>] i =", i, "\tbuf: 0 to", bufEnd, "to", size, \
            #        "\twav:", wavStart, "to", n, "|", (n - wavStart))

            buf[0:bufEnd] = wav[wavStart:n]
            buf[bufEnd:size] = 0.0
        else:
            wavStart = i
            wavEnd = wavStart + size

            #print("[=] i =", i, "\tbuf: 0 to", size, \
            #        "\t\twav:", wavStart, "to", wavEnd, \
            #        "|", (wavEnd - wavStart))

            buf[:] = wav[wavStart:wavEnd]

        if type(wnd) != type(None):
            buf *= wnd

        X = rfft(buf)

        absNorm = np.abs(X) / size
        ab[:, c] = absNorm
        an[:, c] = util.angle(X)
        x[:, c] = X

        c += 1

    assert c == iters
    return ab, an, x


################################################################################
def bmp2wav(fs, l, x, mask, size, overlapDec):
    """Apply a filter mask to a spectrogram image and transform it back to
    wave samples.

    This function cheats by first performing wav2bmp again and applying the
    filter mask onto the resultant complex FFT data, before then converting
    that image into wave samples. This removes the need to convert the
    amplitude and angle BMPs back into complex numbers for the filter mask
    scaling.

    TODO: this function does not amplify the start and end iterations
    appropriately, according to how many times a sample was used in all
    iterations.
    """
    assert x.ndim == 2
    assert x.ndim == mask.ndim
    assert x.shape == mask.shape
    assert x.dtype == complex
    assert mask.dtype == "float32"

    start, n, step, iters = get_fft_stats(l, size, overlapDec)
    fftI = 0
    wavI = start
    out = np.zeros(l, dtype="float32")
    mult = np.log2(1.0 / (1.0 - overlapDec))

    if mult == 0.0:
        mult = 1.0

    print("multiple =", mult)

    while wavI < l:
        assert wavI < n

        buf = irfft(x[:, fftI] * mask[:, fftI]) / mult

        if wavI < 0:
            bufStart = size - (size + wavI)
            bufEnd = size
            wavEnd = wavI + size

            #print("[<] wavI =", wavI, "\tout: 0 to", wavEnd, \
            #        "\tbuf:", bufStart, "to", bufEnd, "|", wavEnd)

            out[0:wavEnd] += buf[bufStart:bufEnd]
        elif (wavI + size) >= n:
            bufEnd = n - wavI
            wavStart = wavI

            #print("[>] wavI =", wavI, "\tout:", wavStart, "to", n, \
            #        "\tbuf: 0 to", bufEnd, "|", (n - wavStart))

            out[wavStart:n] += buf[0:bufEnd]
        else:
            wavStart = wavI
            wavEnd = wavStart + size

            #print("[=] wavI =", wavI, "\tout:", wavStart, "to", wavEnd, \
            #        "\t\tbuf: 0 to", buf.shape[0], "|", (wavEnd - wavStart))

            out[wavStart:wavEnd] += buf[:]

        fftI += 1
        wavI += step

    assert fftI == iters

    return out
