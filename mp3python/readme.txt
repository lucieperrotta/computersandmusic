To encode a WAVE file, run the program with the 
following command line arguments:

python encoder.py inwavfile.wav [outmp3file] bitrate

e.g.

python encoder.py samples/sine.wave samples/sine.mp3 320

for compression with nitrate 320 kbps.

If outmp3file is omitted, same filename as 
input file is used, but with .mp3 extension.

Supported bitrates are (64 kbps to 448 kbps in 32 kbps steps). E.g. 64, 128, 256, 512 kbps, with the lower the bitrate, the higher the compression ratio.

If there is an error reading the WAVE file, maybe 
it is not supported by the WavRead class (currently
only supports standard integer PCM WAVE files).

Two sample WAVE input files, 'sine.wav' - a stereo composed of 440 and 880 Hz sine waves, sampled at 44100 Hz -  and 'ctt.wav' - acappela sample, are given for testing.

To launch notebook, make sure ipython and notebook packages are properly installed, and run the following command,

ipython notebook mp3python.ipynb

To convert an ipython notebook to html format,

ipython nbconvert mp3python.ipynb
