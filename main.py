#!/usr/bin/python3
#
# MIT License
#
# Copyright (c) 2022 Adam Dodd
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
import scipy.signal as sig
import soundfile as sf

#%%#############################################################################

class W2B:
    def __init__(self):
        pass
    
    @classmethod
    def create_from_wav(cls, file_name):
        ret = W2B()
        data, ret.fs = sf.read(file_name)
        ret.data = data.astype(np.float32).T
        
        # TODO

#%%#############################################################################

def main(name, size, overlapDec):
    fs, s, l = wav.read(name)

    # If stereo, take left channel
    if s.ndim == 2:
        s0 = s[:, 0]
    else:
        s0 = s

    print("Computing FFT data...")
    ab, an, x = fft.wav2bmp(fs, s0, size, overlapDec)

    print("Drawing graphs...")
    fig = plt.figure()
    plt.plot(np.arange(0, s0.size), s0)
    plt.show(block=False)

    plot.draw_abs(name, fs, size, overlapDec, ab)
    plot.draw_abs_db(name, fs, size, overlapDec, ab)
    #plot.draw_abs_db_log(name, fs, size, overlapDec, ab)
    plot.draw_ang(name, fs, size, overlapDec, ab, an)

    print("Writing images...")
    img.write_abs(name, fs, size, overlapDec, ab)
    img.write_abs_db(name, fs, size, overlapDec, ab)
    #img.write_abs_db_log(name, fs, size, overlapDec, ab)
    img.write_ang(name, fs, size, overlapDec, ab, an)

    print("Done")
    plt.show()


################################################################################
if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
    else:
        print("Usage: " + sys.argv[0] +
                " <WAV file> <size> <overlap>")
        sys.exit(1)
