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
from numpy.fft import rfft, irfft, rfftfreq

from . import util


################################################################################
def get_pad_iters(size, overlapDec):
    # So, start the spectrogram with size * overlapDec's worth of samples not
    # in the FFT (4 zeros 50%, 6 zeroes 75%, 7 zeroes 87.5%, etc.)

    assert type(size) == int
    assert type(overlapDec) == float

    overlap = int(size * overlapDec)
    step = size - overlap

    if np.floor(step) != step:
        raise ValueError("`overlapDec' must produce whole step")

    sampPerIter = int(size / step)

    # Iterations is going to be one less than the number of times that step
    # goes into size
    padIters = sampPerIter - 1

    return padIters


################################################################################
def get_fft_stats(n, size, overlapDec):
    """Calculate the stats needed to iteratively compute the FFT over a set of
    wave samples with a given length, size and overlap.

    The number of iterations for a perfectly "divisible" (no right padding)
    number of samples `n` by the size and overlapDec is:

        iters = n / (1.0 - overlapDec)

        Add one more for padding if needed.
    """

    assert type(n) == int
    assert type(size) == int
    assert type(overlapDec) == float

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

    if int(sizeLog2) != sizeLog2:
        raise ValueError("Size must be 2^n")

    if np.floor(size * (1.0 - overlapDec)) != (size * (1.0 - overlapDec)):
        raise ValueError("`overlapDec' must produce whole step w.r.t. `size`")

    # Given the size and overlap, the "window" that we consider for each
    # iteration of the spectrogram steps `step` samples along the WAV data
    step = int(size * (1.0 - overlapDec))

    # Calculate left padding
    leftPadIters = get_pad_iters(size, overlapDec)
    start = -(step * leftPadIters)

    # The minimum number of iterations needed (may need one more for padding)
    iters = int(n / step) + leftPadIters
    nModStep = n % step

    # Is right padding needed?
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

    assert type(start) == int
    assert type(step) == int
    assert type(iters) == int

    return start, step, iters


################################################################################
def get_ifft_stats(iters, size, overlapDec):
    """Calculate the stats needed to iteratively compute the IFFT...

    Future enhancements: select size and overlapDec automagically based on
    image dims.
    """

    assert type(iters) == int
    assert type(size) == int
    #assert type(bins) == int

    #size = (bins - 1) * 2
    #log2size = np.log2(size)

    #if np.floor(log2size) != log2size:
    #    raise ValueError("`bins` does not satisfy: (FFTSize/2) + 1")

    if (overlapDec >= 1.0) or (overlapDec < 0.0):
        raise ValueError("Overlap must be LT 1.0, GE 0.0")
    elif overlapDec != 0.0:
        overlapLog2 = np.log2(1.0 / (1.0 - overlapDec))

        if int(overlapLog2) != overlapLog2:
            raise ValueError("Overlap must be 1 / (2^n)")

    paddedIters = get_pad_iters(size, overlapDec)
    nonPaddedIters = int(iters - (paddedIters * 2))

    #overlapScale = 1.0 / (1.0 - overlapDec)
    #n = int(float(size) / overlapScale)
    #padIters = get_pad_iters(size, overlapDec)

    n = int(np.ceil((size * nonPaddedIters) * (1 - overlapDec)))

    return n


################################################################################
def gen_phase_data(fs, size, iters):
    """Generate periodic phase data (not sure if this is suitable)

    XXX: future enhancement: take start and end frequencies.
    """

    assert fs > 0.0
    assert type(size) == int
    assert type(iters) == int

    binFreqs = rfftfreq(size, d=(1.0 / fs))
    out = np.zeros((len(binFreqs), iters), dtype="float32")
    ts = 1.0 / fs
    t = np.array([ts * t for t in range(0, iters)], dtype="float32")
    pi2 = 2.0 * np.pi

    for b in range(0, len(binFreqs)):
        pi2f = pi2 * binFreqs[b]
        out[b] = np.sin(pi2f * t)

    return out


################################################################################
def wav2bmp(fs, wav, size=1024, overlapDec=0.0, window=np.hanning):
    """Transform wave samples into a spectrogram image.

    Warning: using a window in the bmp2wav flow (when recomputing the complex
    FFT result for resynthesis with the mask image) will result in a very badly
    scaled result!

    See tests/test_bmp2wav.py.
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
        raise ValueError("Expected `window' to be a function, NumPy array " +
                "or None")

    start, step, iters = get_fft_stats(l, size, overlapDec)
    c = 0
    ab = np.zeros((fftLen, iters), dtype="float32")
    an = np.zeros((fftLen, iters), dtype="float32")
    x = np.zeros((fftLen, iters), dtype=complex)
    buf = np.ndarray(size, dtype="float32")

    for i in range(start, l, step):
        # Do stuff

        if i < 0:
            bufStart = size - (size + i)
            bufEnd = size
            wavEnd = i + size

            #print("[<] i =", i, "\tbuf: 0 to", bufStart, "to", bufEnd, \
            #        "\twav: 0 to", wavEnd, "|", wavEnd)

            buf[0:bufStart] = 0.0
            buf[bufStart:bufEnd] = wav[0:wavEnd]
        elif (i + size) >= l:
            bufEnd = l - i
            wavStart = i

            #print("[>] i =", i, "\tbuf: 0 to", bufEnd, "to", size, \
            #        "\twav:", wavStart, "to", l, "|", (l - wavStart))

            buf[0:bufEnd] = wav[wavStart:l]
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
    """
    assert x.ndim == 2
    assert x.ndim == mask.ndim
    assert x.shape == mask.shape
    assert x.dtype == complex
    assert mask.dtype == "float32"

    start, step, iters = get_fft_stats(l, size, overlapDec)
    fftI = 0
    wavI = start
    out = np.zeros(l, dtype="float32")
    mult = size / step

    while wavI < l:
        buf = irfft(x[:, fftI] * mask[:, fftI])

        if wavI < 0:
            bufStart = size - (size + wavI)
            bufEnd = size
            wavEnd = wavI + size

            #print("[<] wavI =", wavI, "\tout: 0 to", wavEnd, \
            #        "\tbuf:", bufStart, "to", bufEnd, "|", wavEnd)

            out[0:wavEnd] += buf[bufStart:bufEnd] / mult
        elif (wavI + size) >= l:
            bufEnd = l - wavI
            wavStart = wavI

            #print("[>] wavI =", wavI, "\tout:", wavStart, "to", l, \
            #        "\tbuf: 0 to", bufEnd, "|", (l - wavStart))

            out[wavStart:l] += buf[0:bufEnd] / mult
        else:
            wavStart = wavI
            wavEnd = wavStart + size

            #print("[=] wavI =", wavI, "\tout:", wavStart, "to", wavEnd, \
            #        "\t\tbuf: 0 to", buf.shape[0], "|", (wavEnd - wavStart))

            out[wavStart:wavEnd] += buf[:] / mult

        fftI += 1
        wavI += step

    assert fftI == iters

    return out
