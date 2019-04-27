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

The tools in this repository operate on WAV files that you have on your system. When image files are generated, they will be written to the same directory as the source WAV file. **If you have an SSD**, beware that using large FFT sizes with big overlaps can result in large files being written to disk. I would recommend using this tool on a hard disk drive and, to be sure, copy any WAVs to the repository directory and use them from there.

## Useful tools to check out first

I have provided the tools `print_sizes.py` and `print_overlaps.py` which will help in choosing a suitable size and overlap when analysing a WAV.

## Quick start #1: my first analysis!

Navigate to the `wav2bmp` directory in a command-line terminal. In this example I am using the Command Prompt on Windows. Adjust paths where appropriate, according to where you cloned this repository.

```
C:\Users\Adam>cd C:\wav2bmp

C:\wav2bmp>
```

I have already copied my first WAV file, `pdp_soyboy.wav` to the repository directory.

The script that will generate our BMPs is called `wav2bmp.py`. To see how we should run it, try without any arguments:

```
C:\wav2bmp>python wav2bmp.py
Usage: wav2bmp.py <WAV file> <size> <overlap>

C:\wav2bmp>
```

I've already decided that the most suitable size and overlap values (in my opinion) are `4096` and `0.875`, respectively. Let's run them on the WAV:

```
C:\wav2bmp>python wav2bmp.py pdp_soyboy.wav 4096 0.875
Read WAV: "pdp_soyboy.wav" (fs = 44100, len = 480734)
Computing FFT data...
Drawing graphs...
Writing images...
Writing image file "pdp_soyboy.wav_fs44100_s4096_o0.875_ab_db.bmp"
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Writing raw file "pdp_soyboy.wav_fs44100_s4096_o0.875_ab_db.npy"
```

Don't worry about "lossy conversion" warnings.

A graph will have been drawn and displayed. This is the spectrogram. Note that the axes are purely FFT and frequency bin indices. Converting these values to either time or frequency, respectively, requires more work.

If you're interested, NumPy provides the `numpy.fft.rfftfreq(n)` function to determine the frequencies of each of the specific bins. This is used already in the code as part of the function `util.log_freq()`, used by the logarithmic graph and image functions `w2b.plot.draw_abs_db_log()` and `w2b.img.write_abs_db_log()`, respectively.

As stated in the script output, `wav2bmp.py` has written two files:

1. `pdp_soyboy.wav_fs44100_s4096_o0.875_ab_db.bmp`
2. `pdp_soyboy.wav_fs44100_s4096_o0.875_ab_db.npy`

W2B writes images with several pieces of information encoded into the name:

- Source WAV name
- Sample rate ("fs44100", i.e. 44.1 kHz)
- FFT size ("s4096", i.e. 4096 bins)
- FFT overlap ("o0.875", i.e. the FFT window is moved one-eighth of the legth of the FFT size)
- Type of data ("ab_db", i.e. this is the absolute (amplitude) data, with logarithmic values)

This is purely for your information - W2B scripts don't actually parse this.

## Quick start #2: my first resynthesis!

Now that we have a spectrogram, it's time to create a mask with which to modify the amplitude data. I do this with GIMP.

TODO
