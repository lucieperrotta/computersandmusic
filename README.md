# <i>COM-418 - Computers and Music</i>

<div align="right"><a href="https://people.epfl.ch/lucie.perrotta">Lucie Perrotta</a> and <a href="https://people.epfl.ch/paolo.prandoni">Paolo Prandoni</a>, <a href="https://www.epfl.ch/labs/lcav/">LCAV, EPFL</a></div>

---
This repository is part of the EPFL Master's course *Computers and Music* taught during the fall semester. It contains the following notebooks:
- **[Mp3Encoder](./Mp3Encoder/Mp3Encoder.ipynb)**: A MPEG1-layer 1 encoder, converting *.wav* into *.mp3* files.
- **[PitchScalingAndTimeStretching](./PitchScalingAndTimeStretching.ipynb)**: Methods for independently changing the pitch and length of an audio signal.
- **[Synthesizer](./Synthesizer.ipynb)**: Digital, additive, FM, and wavetable syntheses, and LFO modulation.
- **[Equalizer](./Equalizer.ipynb)**: An audio EQ composed of 3 types of frequency filters (notch, cut, and shelf).

- **[ChannelVocoder](.ChannelVocoder.ipynb)**: A vocoder combining a carrier synthesizer with a modulator voice.
- **[Compressor](./Compressor.ipynb)**: A dynamic range compressor controlled by threshold, ratio, knee with, makeup, attack, and release.
- **[Reverb](./Reverb.ipynb)**: Artificial acoustic reverberation built from FIR and IIR filters.
- **[NonlinearModelling](./NonlinearModelling.ipynb)**: Methods for modelling and identifying dynamic nonlinear systems such as amplifiers.
- **[DeepLearning](./DeepLearning.ipynb)**: Some examples of deep learning applications in production.
- **[Helpers](./Helpers.ipynb)**: A notebook with helpers functions for the others notebooks, along with examples and explanations.

The *[requirements.txt](./requirements.txt)* file contains the libraries needed for running the code.

The *[samples](./samples/)* directory contains *.wav* audio files needed for running the notebooks.
