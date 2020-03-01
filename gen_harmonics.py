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

import w2b.util as util


################################################################################
def main(maskName):

    print("Reading mask image...")
    mask = np.flipud(util.norm(iio.imread(maskName)))
    assert mask.ndim == 2

    print("Drawing harmonics...")
    mask2 = np.array(mask, dtype="float32")
    mask3 = np.array(mask, dtype="float32")
    print("dim: {}".format(mask2.shape))

    fn = mask2.shape[0]
    tn = mask2.shape[1]
    z = np.zeros(fn, dtype="float32")

    print("Time axis length: {}".format(tn))
    print("Bins axis length: {}".format(fn))

    for t in range(0, tn):
        low = -1
        high = -1

        closeArray = np.isclose(z, mask2[:, t])

        for f in range(0, fn):
            if (low == -1) and (~closeArray[f]):
                print("Found lowest at idx {}".format(f))
                low = f
            elif (low != -1) and (high == -1) and closeArray[f]:
                print("Found highest at idx {}". format(f))
                high = f
                break

        if (low != -1) and (high != -1):
            frange = int(high - low)
            frangeDiv2 = int(frange / 2.0)
            fundFreqIdx = low + frangeDiv2
            nextFreqIdx = fundFreqIdx * 2
            print("range (idx): {} to {} ({})".format(low, high, frange))

            while (nextFreqIdx + frangeDiv2) < fn:
                dstStart = nextFreqIdx - frangeDiv2
                dstEnd = nextFreqIdx + frangeDiv2
                srcStart = fundFreqIdx - frangeDiv2
                srcEnd = fundFreqIdx + frangeDiv2

                mask3[dstStart:dstEnd, t] = mask2[srcStart:srcEnd, t]

                nextFreqIdx += fundFreqIdx

    fig = plt.figure()
    fig.suptitle("Mask image vs w/ harmonics")
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray", origin="lower")
    plt.subplot(1, 2, 2)
    plt.imshow(mask3, cmap="gray", origin="lower")
    plt.show(block=False)

    iio.imwrite(maskName + "_harmmask.bmp", np.flipud(mask3))

    print("Done")
    plt.show()


################################################################################
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Usage: " + sys.argv[0] + " <filter BMP>")
        sys.exit(1)
