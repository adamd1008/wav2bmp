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

import argparse
import sys

import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import w2b.fft as fft
import w2b.plot as plot
import w2b.util as util
import w2b.wav as wav


################################################################################
def wavMain(wavName, maskName, size, overlapDec):
    fs, s, l = wav.read(wavName)

    # If stereo, take left channel
    if s.ndim == 2:
        s0 = s[:, 0]
    else:
        s0 = s

    print("Reading mask image...")
    mask = np.flipud(util.norm(iio.imread(maskName)))
    assert mask.ndim == 2

    print("Retrieving FFT data from WAV...")
    # XXX: MUST USE NO WINDOW!
    ab, an, x = fft.wav2bmp(fs, s0, size, overlapDec, window=None)

    # XXX: demonstrating the impact of blank phase data
    x2 = np.ndarray(x.shape, dtype=complex)
    x2[:] = np.real(x) + (1j * np.zeros(an.shape, dtype="float32"))
    #x2[:] = np.real(x) + (1j * fft.gen_phase_data(fs, size, x.shape[1]))
    assert np.allclose(np.real(x), np.real(x2))
    #assert np.allclose(np.imag(x2), np.zeros(x2.shape))

    print("Resynthesizing mask image...")
    out = fft.bmp2wav(fs, l, x2, mask, size, overlapDec)

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
def imgMain(fs, imgName, size, overlapDec):
    assert imgName != None

    print("Reading image...")
    img = np.flipud(util.norm(iio.imread(imgName)))

    if img.ndim == 3:
        # This is an RGB image? Transform into greyscale

        print("RGB image detected; averaging to greyscale...")
        greyImg = np.ndarray(img.shape[0:2], dtype="float32")
        greyImg[:, :] = np.average(img[:, :])
    if img.ndim == 2:
        greyImg = img
    else:
        raise ValueError("This doesn't look like an image!")

    # Resize image here
    bins = int((size / 2) + 1)
    newShape = (bins, greyImg.shape[1])

    # TODO: uncomment this when sure resizing works
    #if greyImg.shape != newShape:
    #    print("Resizing image from {} to {}".format(greyImg.shape, newShape))
    #    pilImage = Image.fromarray(greyImg)
    #    pilImage2 = pilImage.resize(newShape)
    #    greyImg2 = np.asarray(pilImage2)
    #else:
    #    greyImg2 = greyImg

    print("Resizing image from {} to {}".format(greyImg.shape, newShape))
    pilImage = Image.fromarray(greyImg)
    # PIL needs the new dims transposed
    pilImage2 = pilImage.resize((newShape[1], newShape[0]))
    greyImg2 = np.asarray(pilImage2)

    iters = greyImg2.shape[1]

    # Angle data
    print("Generating blank phase data...")
    #an = np.zeros(greyImg2.shape, dtype="float32")
    an = fft.gen_phase_data(fs, size, iters)

    plot.draw_abs("gen_phase_data", fs, size, overlapDec, an, block=True)

    # Mask data
    print("Generating blank mask data...")
    mask = np.ones(greyImg2.shape, dtype="float32")

    combinedX = np.ndarray(greyImg2.shape, dtype=complex)
    combinedX[:] = greyImg2 + (1j * an)
    assert combinedX.dtype == complex
    assert np.allclose(np.real(combinedX), greyImg2)

    n = fft.get_ifft_stats(iters, size, overlapDec)

    print("New WAV sample count: {}".format(n))

    out = fft.bmp2wav(fs, n, combinedX, mask, size, overlapDec)
    #out = util.norm(fft.bmp2wav(fs, n, combinedX, mask, size, overlapDec))
    #wavName = util.gen_filename(imgName, fs, size, overlapDec, "wav")
    wavName = imgName + ".wav"
    wav.write(wavName, fs, out)

    print("Drawing WAV graph...")
    fig = plt.figure()
    plt.plot(np.arange(0, out.size), out)
    plt.show()


################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Image resynthesis engine",
            epilog="Note: a mask image is applicable only when processing "
            + "a WAV")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--wav",
            type=str, help="WAV file to get FFT data from")
    group.add_argument("-i", "--image",
            type=str, help="BMP image to resynthesize")

    parser.add_argument("-m", "--mask",
            type=str, help="BMP image mask to use as filter")
    parser.add_argument("-f", "--fs",
            type=int, help="Sample rate")

    parser.add_argument("size",
            type=int, help="FFT size")
    parser.add_argument("overlapDec",
            type=float, help="iteration overlap (decimal)")

    args = parser.parse_args()

    wavName = args.wav
    imgName = args.image
    maskName = args.mask
    fs = args.fs
    size = args.size
    overlapDec = args.overlapDec

    if (imgName != None) and (maskName != None):
        raise ValueError(
                "Mask image is not applicable to pure image resynthesis")

    if imgName != None:
        if fs == None:
            fs = 44100

        imgMain(fs, imgName, size, overlapDec)
    else:
        if fs != None:
            raise ValueError("Sample rate not applicable")

        wavMain(wavName, maskName, size, overlapDec)

