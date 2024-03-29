# <i>COM-418 - Computers and Music</i>

<div align="right"><a href="https://people.epfl.ch/lucie.perrotta">Lucie Perrotta</a> and <a href="https://people.epfl.ch/paolo.prandoni">Paolo Prandoni</a>, <a href="https://www.epfl.ch/labs/lcav/">LCAV, EPFL</a></div>

---
This repository is part of the EPFL Master's course *Computers and Music* taught during the spring semester. The course's lecture notes can be found [on this link](https://www.overleaf.com/read/ngrypxzshdqd). The lecture notes are supported by Jupyter notebooks to illustrate practical examples of the concepts taught in class. This repository contains the following notebooks:
- **[DAC](./DAC.ipynb)**: Simple visualizations of the building bricks of an ADC and a DAC.
- **[OneBitMusic](./OneBitMusic.ipynb)**: Encoding music over 1bit-samples using oversampling and sigma-delta.
- **[QuantizationNoise](./QuantizationNoise.ipynb)**: Demonstration of the Tsividis' paradox when quantizing a signal.
- **[Mp3Encoder](./Mp3Encoder/Mp3Encoder.ipynb)**: A MPEG1-layer 1 encoder, converting *.wav* into *.mp3* files.
- **[PitchScalingAndTimeStretching](./PitchScalingAndTimeStretching.ipynb)**: Methods for independently changing the pitch and length of an audio signal.
- **[ChannelVocoder](./ChannelVocoder.ipynb)**: A vocoder combining a carrier synthesizer with a modulator voice.
- **[Synthesizer](./Synthesizer.ipynb)**: Digital, additive, FM, and wavetable syntheses, and LFO modulation.
- **[PhysicalModelling](./PhysicalModelling.ipynb)**: Physical modelling implementations of different instruments such as strings or percussions.
- **[Equalizer](./Equalizer.ipynb)**: An audio EQ composed of 3 types of frequency filters (notch, cut, and shelf).
- **[Compressor](./Compressor.ipynb)**: A dynamic range compressor controlled by threshold, ratio, knee with, makeup, attack, and release.
- **[Reverb](./Reverb.ipynb)**: Artificial acoustic reverberation built with FIR and IIR filters.
- **[Beatles](./Beatles.ipynb)**: A simple implementation of a guitar+amplifier setup featuring _the Beatles_.
- **[NonlinearModelling](./NonlinearModelling.ipynb)**: Methods for modelling and identifying dynamic nonlinear systems such as amplifiers.
- **[DeepLearning](./DeepLearning.ipynb)**: Some examples of deep learning applications in audio production.
- **[Helpers](./Helpers.ipynb)**: A notebook with helpers functions for the others notebooks, along with examples and explanations.

The *[requirements.txt](./requirements.txt)* file contains the libraries needed for running the code.

The *[samples](./samples/)* directory contains *.wav* audio files needed for running the notebooks.

The *[pictures](./pictures/)* directory contains the pictures needed for running the notebooks.
