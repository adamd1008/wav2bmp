float complexAbs(const float re, const float im)
{
    return sqrt(pow(re, 2.0f) + pow(im, 2.0f));
}

float complexAngAtan2(const float re, const float im)
{
    const float pi2 = 2.0f * M_PI_F;
    float ret = atan2(im, re);

    if (ret < 0.0f)
        ret += pi2;

    return ret;
}

// Chinese hat smoothing for angle (for  A E S T H E T I C  purposes only!)
float complexAngAtan2CHat(const float re, const float im)
{
    const float pi2 = M_PI_F * 2.0f;
    const float cAng = complexAngAtan2(re, im) * 2.0f;
    float ret = cAng;

    if (cAng > pi2)
    {
        ret = pi2 - (cAng - pi2);
    }

    return ret;
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

    //float thisAng = atan2(im, re) + M_PI_F;
    float thisAng = complexAngAtan2(re, im);
    //float thisAng = complexAngAtan2CHat(re, im);

    abs[binIdx] = thisAbs;
    ang[binIdx] = thisAng;
}
