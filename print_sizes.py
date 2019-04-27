#!/usr/bin/python3
#
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


################################################################################
def main():
    n = 8

    while n <= 16:
        x = int(np.power(2.0, n))
        print("n = " + str(n) + "\tsize = " + str(x))
        n += 1

    print("""
FFT size is a tradeoff between the frequency resolution and the \
time resolution (when they are appended and combined into a spectrogram). If \
you pick a high size (>4096) you will notice that one column of the \
spectrogram represents enough time that transients (sounds with a drastic \
change in frequency in a short time) may be too hard to see, and so modifying \
them is not possible.

For typical audio data, I prefer 1024, 2048 or 4096. But, by all means, \
experiment!

W2B supports all FFT sizes, I think. But beware that large sizes, in \
conjunction with high overlaps, may take a long time and require a *huge* \
amount of memory.""")


################################################################################
if __name__ == "__main__":
    main()
