/**
 * MIT License
 *
 * Copyright (c) 2020 Adam Dodd
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

float complexAbs(const float re, const float im)
{
    return sqrt(pow(re, 2.0f) + pow(im, 2.0f));
}

/* Compute normalised angle data (0.0 <= ret < 1.0) */
float complexAngAtan2(const float re, const float im)
{
    const float pi2 = 2.0f * M_PI_F;
    float ret = atan2(im, re);

    if (ret < 0.0f)
        ret += pi2;

    return ret / pi2;
}

__kernel void slowft(
        const uint fs,
        __global const float samp[],
        const uint size,
        const uint bins,
        const uint iters,
        __global const float freqs[],
        const uint freqCount,
        __global float abs[],
        __global float ang[])
{
    const float pi2 = 2.0f * M_PI_F;
    uint iter = get_global_id(0);
    uint bin = get_global_id(1);

    uint binIdx = (bin * iters) + iter;
    float fps = (pi2 * freqs[bin]) / ((float) fs);
    float re = 0.0f;
    float im = 0.0f;

    for (uint n = 0; n < size; n++)
    {
        uint sampIdx = (n * iters) + iter;
        float fpsn = fps * ((float) n);

        re += samp[sampIdx] * cos(fpsn);
        im -= samp[sampIdx] * sin(fpsn);
    }

    re /= ((float) size);
    im /= ((float) size);

    float thisAbs = complexAbs(re, im);
    float thisAng = complexAngAtan2(re, im);

    abs[binIdx] = thisAbs;
    ang[binIdx] = thisAng;
}
