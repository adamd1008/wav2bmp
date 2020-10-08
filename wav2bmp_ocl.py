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

import matplotlib.pyplot as plt
import numpy as np

import w2b.ft_ocl as ft_ocl
import w2b.img as img
import w2b.plot as plot
import w2b.wav as wav


################################################################################
def main(name, size, bins, startFreq, endFreq, overlapDec):
    fs, s, l = wav.read(name)

    # If stereo, take left channel
    if s.ndim == 2:
        s0 = s[:, 0]
    else:
        s0 = s

    print("Computing FFT data...")
    ab, an = ft_ocl.wav2bmp_ocl(
            fs, s0, size, bins, startFreq, endFreq, overlapDec)

    print("Drawing graphs...")
    fig = plt.figure()
    plt.plot(np.arange(0, s0.size), s0)
    plt.show(block=False)

    plot.draw_abs(name, fs, size, overlapDec, ab, block=False)
    plot.draw_abs_db(name, fs, size, overlapDec, ab, block=False)
    plot.draw_ang(name, fs, size, overlapDec, ab, an,
            bins, startFreq, endFreq, block=False)

    print("Writing images...")
    #img.write_abs(name, fs, size, overlapDec, ab, bins, startFreq, endFreq)
    #img.write_abs_db(name, fs, size, overlapDec, ab,
    #        bins, startFreq, endFreq)
    img.write_ang(name, fs, size, overlapDec, ab, an, bins, startFreq, endFreq)

    print("Done")
    plt.show()


################################################################################
if __name__ == "__main__":
    if len(sys.argv) == 7:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    else:
        print("Usage: " + sys.argv[0] +
                " <WAV file> <size> <bins> <startFreq> <endFreq> <overlap>")
        sys.exit(1)
