# wav2bmp (W2B)

A simple Python 3 library for converting a WAV file into a BMP (and back!).

## Install prerequisites

### Download and install Python 3

Download the latest version of Python 3.X from the [official website](https://www.python.org/downloads/).

When installing, be sure to tick the box titled "Add Python 3.X to PATH". This will make it much easier to invoke Python from a command-line terminal.

### Install the required Python 3 libraries

From the command-line:

```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose imageio
```

Note: you may need to use the command `python3` in place of `python`, depending on your platform. If in doubt, to find the version you are using, use command `python --version`.

### Clone this repository

From the command-line:

```
git clone https://github.com/adamd1008/wav2bmp.git
```

It's best to clone the repository to a hard disk drive. More details on that below.

### (Optional) Install an image editor

This is up to you, if you want to perform resynthesis (i.e. BMP -> WAV). I use GIMP in my example below.

### Copy any WAVs to the repository directory

One WAV, `square_2.wav`, is included as part of this repository. It is a 500 Hz square wave with only two harmonics, and has a sample rate of 10 kHz. As such, it's an extremely simple example that helps to illustrate resynthesis. Be warned: it's normalised, so will be very loud!

The tools in this repository operate on WAV files that you have on your system. When image files are generated, they will be written to the same directory as the source WAV file. **If you have an SSD**, beware that using large FFT sizes with big overlaps can result in large files being written to disk. I would recommend using this tool on a hard disk drive and, to be sure, copy any WAVs to the repository directory and use them from there.

## Useful tools to check out first

I have provided the tools `print_sizes.py` and `print_overlaps.py` which will help in choosing a suitable size and overlap when analysing a WAV.

## Quick start #1: my first analysis!

Navigate to the `wav2bmp` directory in a command-line terminal. In this example I am using the Command Prompt on Windows. Adjust paths where appropriate, according to where you cloned this repository.

```
C:\Users\Adam>cd /d D:\Work\wav2bmp
```

In this first example we will be using the included WAV called `square_2.wav`.

The script that will generate our BMPs is called `wav2bmp.py`. To see how we should run it, try without any arguments:

```
C:\wav2bmp>python wav2bmp.py
Usage: wav2bmp.py <WAV file> <size> <overlap>
```

I've already decided that the most suitable size and overlap values (in my opinion) are `1024` and `0.5`, respectively. Let's run them on the WAV:

```
D:\Work\wav2bmp>python wav2bmp.py square_2.wav 1024 0.5
Read WAV: "square_2.wav" (fs = 10000, len = 30000)
Computing FFT data...
Drawing graphs...
Writing images...
Writing image file "square_2.wav_fs10000_s1024_o0.5_ab_db.bmp"
WARNING:root:Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Writing raw file "square_2.wav_fs10000_s1024_o0.5_ab_db.npy"
```

Don't worry about "lossy conversion" warnings.

A graph will have been drawn and displayed. This is the spectrogram. Note that the axes are purely FFT and frequency bin indices. Converting these values to either time or frequency, respectively, requires more work.

If you're interested, NumPy provides the `numpy.fft.rfftfreq(n)` function to determine the frequencies of each of the specific bins. This is used already in the code as part of the function `util.log_freq()`, used by the logarithmic graph and image functions `w2b.plot.draw_abs_db_log()` and `w2b.img.write_abs_db_log()`, respectively.

As stated in the script output, `wav2bmp.py` has written two files:

1. `square_2.wav_fs10000_s1024_o0.5_ab_db.bmp`
2. `square_2.wav_fs10000_s1024_o0.5_ab_db.npy`

Ignore the `.npy` file for now. W2B writes images with several pieces of information encoded into the name:

- Source WAV name
- Sample rate ("fs10000", i.e. 10 kHz)
- FFT size ("s1024", i.e. 1024 frequency bins)
- FFT overlap ("o0.5", i.e. the FFT window is moved one-half of the length of the FFT size)
- Type of data ("ab_db", i.e. this is the absolute (amplitude) data, with logarithmic values)

This is purely for your information - W2B scripts don't actually parse this.

## Quick start #2: my first resynthesis!

Now that we have a spectrogram, it's time to create a mask with which to modify the amplitude data. I do this with GIMP. You can use any tool that you like, but the image you create must be grayscale, like the image that was created by `wav2bmp.py`, and have the exact same dimensions.

Using GIMP, I load the image `square_2.wav_fs10000_s1024_o0.5_ab_db.bmp` and create a new layer. Ensure that you tick the "lock position and size" box, set the opacity to 20-30, and set "fill with" to white. This allows us to draw in black on the new layer while being able to see the spectrogram underneath. Any area of black drawn on the new layer, once exported as its own BMP, can be used with the `bmp2wav.py` script to cancel out the frequencies present in those positions.

We are going to create two mask images. In the first, we are going to draw black along the lower line, thus being left with only the higher harmonic.

Before exporting, click the eye icon in the layers widget to hide the spectrogram, then export the image. I typically use the name of the spectrogram image and add "\_mask" before the file extension. So now I have the mask saved as `square_2.wav_fs10000_s1024_o0.5_ab_db_mask1.bmp`.

Now I show the spectrogram layer again and undo changes to get back to a blank layer. Repeat the above process to draw over the higher harmonic, and export again in the usual way (in my case, with "\_mask2" in the name).

I have included example masks that I made in this repository.

Now we will run the script twice to resynthesize the square wave with each harmonic removed. For the first mask:

```
D:\Work\wav2bmp>python bmp2wav.py square_2.wav square_2.wav_fs10000_s1024_o0.5_ab_db_mask1.bmp 1024 0.5
Read WAV: "square_2.wav" (fs = 10000, len = 30000)
Reading mask image...
Retrieving FFT data from WAV...
Resynthesizing mask image...
multiple = 1.0
min(out) = -0.5746815204620361
max(out) = +0.5729556679725647
Drawing graphs...
Computing spectrogram of the resynthesized WAV...
Drawing more graphs...
Writing WAVs...
Writing WAV: "square_2.wav_fs10000_s1024_o0.5_ab_db_mask1.bmp_in.wav" (fs = 10000, len = 30000)
Writing WAV: "square_2.wav_fs10000_s1024_o0.5_ab_db_mask1.bmp_out.wav" (fs = 10000, len = 30000)
Done
```

Examine and close the drawn graphs. Now for the second:

```
D:\Work\wav2bmp>python bmp2wav.py square_2.wav square_2.wav_fs10000_s1024_o0.5_ab_db_mask2.bmp 1024 0.5
Read WAV: "square_2.wav" (fs = 10000, len = 30000)
Reading mask image...
Retrieving FFT data from WAV...
Resynthesizing mask image...
multiple = 1.0
min(out) = -1.1053712368011475
max(out) = +1.101826548576355
Resynthesized WAV samples are out-of-range; normalising...
Drawing graphs...
Computing spectrogram of the resynthesized WAV...
Drawing more graphs...
Writing WAVs...
Writing WAV: "square_2.wav_fs10000_s1024_o0.5_ab_db_mask2.bmp_in.wav" (fs = 10000, len = 30000)
Writing WAV: "square_2.wav_fs10000_s1024_o0.5_ab_db_mask2.bmp_out.wav" (fs = 10000, len = 30000)
Done
```

And we're done. The two resynthesized WAVs are:

- `square_2.wav_fs10000_s1024_o0.5_ab_db_mask1.bmp_out.wav`
- `square_2.wav_fs10000_s1024_o0.5_ab_db_mask2.bmp_out.wav`

The script also generates a "bmp_in" WAV for easy comparison. This is useful if the source WAV has multiple channels, in which case the "bmp_in" WAV only includes the first channel (which the `wav2bmp.py` script works on).

## Known issues

There is an issue with the amplitude of resynthesized data, especially the start and end samples when an overlap is used.
