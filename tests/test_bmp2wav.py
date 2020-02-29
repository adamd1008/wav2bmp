#!/usr/bin/python3

import re
import unittest
import matplotlib.pyplot as plt
import numpy as np

import w2b.fft as fft
import w2b.util as util
import w2b.wav as wav


################################################################################
class TestBmp2Wav(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.show()


    def draw_diff(self, title, i, o, doAbs=True):
        if doAbs:
            diff = np.abs(i) - np.abs(o)
        else:
            diff = i - o

        fig = plt.figure()
        fig.suptitle("Array differences ({})".format(title))
        plt.plot(np.arange(0, diff.size, dtype="float32"), diff)
        plt.grid(True)
        plt.show(block=False)


    def test_bmp2wav_basic_low_fs(self):
        fs = 100.0
        size = 4
        overlapDec = 0.75
        l = 10
        assert int(l / 2) == (l / 2)

        # Generate a test wav (sequence of +1.0, -1.0, +1.0, -1.0, ...)
        ar = np.array([ 1.0, -1.0], dtype="float32")
        ar2 = np.tile(ar, int(l / 2))
        ab, an, x = fft.wav2bmp(fs, ar2, size, overlapDec, window=None)

        # Generate a blank mask
        mask = np.ndarray(ab.shape, dtype="float32")
        mask[:, :] = 1.0

        # Do b2w
        out = fft.bmp2wav(fs, l, x, mask, size, overlapDec)

        #fig = plt.figure()
        #fig.suptitle("Original WAV vs reynthesized WAV (basic)")
        #plt.subplot(2, 1, 1)
        #plt.plot(np.arange(0, out.size, dtype="float32"), ar2)
        #plt.grid(True)
        #plt.subplot(2, 1, 2)
        #plt.plot(np.arange(0, out.size, dtype="float32"), out)
        #plt.grid(True)
        #plt.show(block=False)
        #self.draw_diff("basic", ar2, out)

        self.assertTrue(np.allclose(ar2, out))


    def test_bmp2wav_basic_high_fs(self):
        fs = 44100.0
        size = 1024
        overlapDec = 0.9375
        l = 100000
        assert int(l / 2) == (l / 2)

        # Generate a test wav (sequence of +1.0, -1.0, +1.0, -1.0, ...)
        ar = np.array([ 1.0, -1.0], dtype="float32")
        ar2 = np.tile(ar, int(l / 2))
        ab, an, x = fft.wav2bmp(fs, ar2, size, overlapDec, window=None)

        # Generate a blank mask
        mask = np.ndarray(ab.shape, dtype="float32")
        mask[:, :] = 1.0

        # Do b2w
        out = fft.bmp2wav(fs, l, x, mask, size, overlapDec)

        #fig = plt.figure()
        #fig.suptitle("Original WAV vs reynthesized WAV (basic)")
        #plt.subplot(2, 1, 1)
        #plt.plot(np.arange(0, out.size, dtype="float32"), ar2)
        #plt.grid(True)
        #plt.subplot(2, 1, 2)
        #plt.plot(np.arange(0, out.size, dtype="float32"), out)
        #plt.grid(True)
        #plt.show(block=False)
        #self.draw_diff("basic", ar2, out)

        self.assertTrue(np.allclose(ar2, out))


    def test_bmp2wav_square_2(self):
        size = 4096
        overlapDec = 0.875

        fs, ar, l = wav.read("square_2.wav")

        if ar.ndim == 2:
            ar2 = ar[:, 0]
        else:
            ar2 = ar

        ab, an, x = fft.wav2bmp(fs, ar2, size, overlapDec, window=None)

        # Generate a blank mask
        mask = np.ndarray(ab.shape, dtype="float32")
        mask[:, :] = 1.0

        # Do b2w
        out = fft.bmp2wav(fs, l, x, mask, size, overlapDec)

        #fig = plt.figure()
        #fig.suptitle("Original WAV vs reynthesized WAV (basic)")
        #plt.subplot(2, 1, 1)
        #plt.plot(np.arange(0, out.size, dtype="float32"), ar2)
        #plt.grid(True)
        #plt.subplot(2, 1, 2)
        #plt.plot(np.arange(0, out.size, dtype="float32"), out)
        #plt.grid(True)
        #plt.show(block=False)
        #self.draw_diff("square_2", ar2, out)

        self.assertTrue(np.allclose(ar2, out))


################################################################################
if __name__ == "__main__":
    unittest.main()
