# wav2bmp

A simple Python 3 library for converting a WAV file into a BMP (and back!).

## Install prerequisites

1. Download the latest version of Python 3 from the [official website](https://www.python.org/downloads/).

2. Install the necessary Python 3 libraries

From the command-line:

```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose imageio
```

Note: you may need to use the command `python3` in place of `python`, depending on your platform. If in doubt, use command `python --version`.

3. Clone this respository

From the command-line:

```
git clone https://github.com/adamd1008/wav2bmp.git
```

4. (Optional) Install an image editor

This is up to you, if you want to perform resynthesis (i.e. BMP -> WAV). I use GIMP in my example below.

## Quick start: my first resynthesis!

### Before we get started

The tools in this repository operate on WAV files that you have on your system. When image files are generated, they will be written to the same directory as the source WAV file. *If you have an SSD*, beware that using large FFT sizes with big overlaps can result in large files being written to disk. I would recommend using this tool on a hard disk drive and, to be sure, copy any WAVs to the repository directory and use them from there.

Navigate to the `wav2bmp` directory in a command-line terminal. In this example I am using the Command Prompt on Windows. Adjust paths where appropriate, according to where you cloned this repository.

```
C:\Users\Adam>cd /d D:\Work\wav2bmp

D:\Work\wav2bmp>
```

I have already copied my first WAV file, `pdp_soyboy.wav` to the repository directory.

TODO
