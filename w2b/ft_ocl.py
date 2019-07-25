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

import os.path
import sys
import types

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

from . import fft
from . import wav


################################################################################
def ft_freqs(bins, startFreq, endFreq):
    if endFreq <= startFreq:
        raise ValueError("`endFreq <= startFreq`")

    assert type(startFreq) == float
    assert type(endFreq) == float

    freqRange = endFreq - startFreq
    freqStep = freqRange / float(bins - 1)

    print("freqRange =", freqRange)
    print("freqStep  =", freqStep)

    return [startFreq + (float(i) * freqStep) for i in range(0, bins)]


################################################################################
def wav2bmp_ocl(fs, wav, size, bins, startFreq, endFreq,
        overlapDec=0.0, window=np.hanning):
    global slowft

    freqs = ft_freqs(bins, startFreq, endFreq)
    freqCount = len(freqs)
    start, n, step, iters = fft.get_fft_stats(len(wav), size, overlapDec)

    platforms = cl.get_platforms()

    if len(platforms) > 1:
        p = -1

        for i in range(0, len(platforms)):
            if "NVIDIA" in platforms[i].vendor:
                p = i
                break

        if p == -1:
            for i in range(0, len(platforms)):
                if "AMD" in platforms[i].vendor:
                    p = i
                    break

        if p == -1:
            print("Unable to intelligently choose a platform; " +
                "defaulting to platform [0]")
            p = 0
    elif len(platforms) == 1:
        p = 0
    else:
        raise RuntimeError("No OpenCL platforms found")

    print(("\nSelected OpenCL platform [{}]\n" +
        "> name: \"{}\"\n" +
        "> version: \"{}\"\n").format(
            p, platforms[p].name, platforms[p].version))

    ctx = cl.Context(
            dev_type=cl.device_type.ALL,
            properties=[(cl.context_properties.PLATFORM, platforms[p])])
    prog = cl.Program(
            ctx, open(os.path.join(
                os.path.dirname(sys.argv[0]), "w2b", "ft_kernel.cl")) \
                        .read()).build()

    print("bins:", bins)
    print("iters:", iters)

    inSamp  = np.ascontiguousarray(np.ndarray((size, iters), dtype="float32"))
    inFreqs = np.ascontiguousarray(np.ndarray(freqCount, dtype="float32"))
    outAbs = np.ascontiguousarray(np.ndarray((bins, iters), dtype="float32"))
    outAng = np.ascontiguousarray(np.ndarray((bins, iters), dtype="float32"))

    inFreqs[:] = freqs

    c = 0
    wnd = np.hanning(size)

    for i in range(start, n, step):
        # Do stuff

        if i < 0:
            bufStart = size - (size + i)
            bufEnd = size
            wavEnd = i + size

            inSamp[0:bufStart, c] = 0.0
            inSamp[bufStart:bufEnd, c] = wav[0:wavEnd]
        elif (i + size) >= n:
            bufEnd = n - i
            wavStart = i

            inSamp[0:bufEnd, c] = wav[wavStart:n]
            inSamp[bufEnd:size, c] = 0.0
        else:
            wavStart = i
            wavEnd = wavStart + size

            inSamp[:, c] = wav[wavStart:wavEnd]

        if type(wnd) != type(None):
            inSamp[:, c] *= wnd

        c += 1

    assert c == iters
    assert not np.any(np.isnan(inSamp))
    assert not np.any(np.isnan(inFreqs))

    print("inSamp.nbytes  =", inSamp.nbytes)
    print("inFreqs.nbytes =", inFreqs.nbytes)
    print("outAbs.nbytes  =", outAbs.nbytes)
    print("outAng.nbytes  =", outAng.nbytes)

    inSampBuf  = cl.Buffer(ctx, mf.READ_ONLY, inSamp.nbytes)
    inFreqsBuf = cl.Buffer(ctx, mf.READ_ONLY, inFreqs.nbytes)
    outAbsBuf  = cl.Buffer(ctx, mf.WRITE_ONLY, outAbs.nbytes)
    outAngBuf  = cl.Buffer(ctx, mf.WRITE_ONLY, outAng.nbytes)

    queue = cl.CommandQueue(ctx)

    cl.enqueue_copy(queue, inSampBuf, inSamp)
    cl.enqueue_copy(queue, inFreqsBuf, inFreqs)
    queue.finish()

    prog.slowft(queue, (iters, bins), None,
            np.uint32(fs), inSampBuf, np.uint32(size), np.uint32(bins),
            np.uint32(iters), inFreqsBuf, np.uint32(freqCount),
            outAbsBuf, outAngBuf)
    queue.finish()

    cl.enqueue_copy(queue, outAbs, outAbsBuf)
    cl.enqueue_copy(queue, outAng, outAngBuf)
    queue.finish()

    assert not np.any(np.isnan(outAbs))
    assert not np.any(np.isnan(outAng))

    print("\nminmax outAbs: {} , {}".format(np.amin(outAbs), np.amax(outAbs)))
    print("minmax outAng: {} , {}".format(np.amin(outAng), np.amax(outAng)))

    return outAbs, outAng
