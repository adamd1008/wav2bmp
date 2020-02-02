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

import numpy as np


################################################################################
def main():
    n = 0

    while n <= 12:
        x = np.power(2.0, n)
        print("n = " + str(n) + "\toverlapDec = " + str(1.0 - (1.0 / x)))
        n += 1

    print("""
Overlap is useful when using bigger FFT sizes. Generally speaking, using a \
higher overlap makes higher FFT sizes clearer. Here are the best combinations \
of size and overlap that I personally find clearest:

FFT size | Suitable overlaps
---------|---------------------------------
     256 | 0.5
     512 | 0.5, 0.75
    1024 | 0.5, 0.75, 0.875
    2048 | 0.75, 0.875
    4096 | 0.875, 0.9375

I find that, for audio, sizes 8192 and up are basically useless regardless of \
the chosen overlap.

Warning: each overlap requires much more time and space than the previous one! \
I wouldn't recommend venturing further than 0.96875 if you have 16 GB or \
less memory.""")


################################################################################
if __name__ == "__main__":
    main()
