#!/usr/bin/python3
#
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

import sys

import imageio as iio
import matplotlib.pyplot as plt
import numpy as np

import w2b.fft as fft
import w2b.plot as plot
import w2b.util as util
import w2b.wav as wav


################################################################################
def main(wavName, maskName, size, overlapDec):
    fs, s, l = wav.read(wavName)

    # If stereo, take left channel
    if s.ndim == 2:
        s0 = s[:, 0]
    else:
        s0 = s

    print("Reading mask image...")
    mask = np.flipud(util.norm(iio.imread(maskName)))
    assert mask.ndim == 2

    maskMin = np.amin(mask)
    maskMax = np.amax(mask)
    print("min(mask) = {:+}".format(maskMin))
    print("max(mask) = {:+}".format(maskMax))

    print("Retrieving FFT data from WAV...")
    # XXX: MUST USE NO WINDOW!
    ab, an, x = fft.wav2bmp(fs, s0, size, overlapDec, window=None)

    print("Resynthesizing FFT data using mask...")
    out = fft.bmp2wav(fs, l, x, mask, size, overlapDec)

    outMin = np.amin(out)
    outMax = np.amax(out)
    print("min(out) = {:+}".format(outMin))
    print("max(out) = {:+}".format(outMax))

    if outMin < -1.0 or outMax > 1.0:
        print("Resynthesized WAV samples are out-of-range; normalising...")
        out2 = util.norm(out)
    else:
        out2 = out

    print("Drawing graphs...")
    fig = plt.figure()
    fig.suptitle("Original WAV vs reynthesized WAV")
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, out2.size, dtype="float32"), s0)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, out2.size, dtype="float32"), out2)
    plt.grid(True)
    plt.show(block=False)

    # XXX: there is a bug(?) which shows the mask image as black even if it's
    # entirely white. WTF?
    fig = plt.figure()
    fig.suptitle("Mask image")
    plt.imshow(mask, cmap="gray", origin="lower")
    plt.show(block=False)

    print("Computing spectrogram of the resynthesized WAV...")
    ab, an, x = fft.wav2bmp(fs, out2, size, overlapDec)

    print("Drawing more graphs...")
    plot.draw_abs("out", fs, size, overlapDec, ab)
    plot.draw_abs_db("out", fs, size, overlapDec, ab)
    plot.draw_ang("out", fs, size, overlapDec, ab, an)

    print("Writing WAVs...")
    wav.write(maskName + "_in.wav", fs, s0)
    wav.write(maskName + "_out.wav", fs, out2)

    print("Done")
    plt.show()


################################################################################
if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    else:
        print("Usage: " + sys.argv[0] + \
                " <WAV file> <Filter BMP> <size> <overlap>")
        sys.exit(1)
