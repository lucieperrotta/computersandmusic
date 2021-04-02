"""
Code by Ivor Rendulic, 2013.
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
# file with functions used in iPython notebook, such as program execution, plotting etc.

import matplotlib.pyplot as plt
import numpy as np
import psychoacoustic
from parameters import *
from common import *


def encode(
    input_buffer,
    params,
    outmp3file,
    **kwargs
    ):
    """Encode the rest of the file. If uniform=true, another file with uniform quantization is created."""

    uniform = kwargs.get('uniform', False)
    if uniform:
        params_uniform = EncoderParameters(input_buffer.fs,
                input_buffer.nch, params.bitrate)
        uniform_bit_allocation = np.zeros((params.nch, N_SUBBANDS),
                dtype='uint8')
        for ch in range(params.nch):
            uniform_bit_allocation[ch, :] = \
                psychoacoustic.smr_bit_allocation(params,
                    np.zeros(N_SUBBANDS))

  # Read baseband filter samples
    baseband_filter = filter_coeffs()

  # Allocate space for 32 subband filters of length 512.
    filterbank = np.zeros((N_SUBBANDS, FRAME_SIZE), dtype='float32')

  # Perform modulation.
    for sb in range(N_SUBBANDS):
        for n in range(FRAME_SIZE):
            filterbank[sb, n] = baseband_filter[n] * np.cos((2 * sb
                    + 1) * (n - 16) * np.pi / 64)

    subband_samples = np.zeros((params.nch, N_SUBBANDS,
                               FRAMES_PER_BLOCK), dtype='float32')

  # Main loop, executing until all samples have been processed.
    while input_buffer.nprocessed_samples < input_buffer.nsamples:

    # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
        for frm in range(FRAMES_PER_BLOCK):
            samples_read = input_buffer.read_samples(SHIFT_SIZE)

      # If all samples have been read, perform zero padding.
            if samples_read < SHIFT_SIZE:
                for ch in range(params.nch):
                    input_buffer.audio[ch].insert(np.zeros(SHIFT_SIZE
                            - samples_read))

      # Filtering = dot product with reversed buffer.
            for ch in range(params.nch):
                subband_samples[ch, :, frm] = np.dot(filterbank,
                        input_buffer.audio[ch].reversed())

    # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
        scfindices = np.zeros((params.nch, N_SUBBANDS), dtype='uint8')
        subband_bit_allocation = np.zeros((params.nch, N_SUBBANDS),
                dtype='uint8')

    # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although
    # scaling is done later, its result is necessary for the psychoacoustic model and calculation of
    # sound pressure levels.
        for ch in range(params.nch):
            scfindices[ch, :] = get_scalefactors(subband_samples[ch, :,
                    :], params.table.scalefactor)
            subband_bit_allocation[ch, :] = \
                psychoacoustic.model1(input_buffer.audio[ch].ordered(),
                    params, scfindices)

    # Scaling subband samples with determined scalefactors.
        for ind in range(FRAMES_PER_BLOCK):
            subband_samples[:, :, ind] /= \
                params.table.scalefactor[scfindices]

        if uniform:
            subband_samples_uniform = np.copy(subband_samples)

    # Subband samples quantization. Multiplication with coefficients 'a' and adding coefficients 'b' is
    # defined in the ISO standard.
        subband_samples_quantized = subband_samples
        for ch in range(params.nch):
            for sb in range(N_SUBBANDS):
                if subband_bit_allocation[ch, sb] != 0:
                    subband_samples[ch, sb, :] *= \
                        params.table.qca[subband_bit_allocation[ch, sb]
                            - 2]
                    subband_samples[ch, sb, :] += \
                        params.table.qcb[subband_bit_allocation[ch, sb]
                            - 2]
                    subband_samples[ch, sb, :] *= 1 \
                        << subband_bit_allocation[ch, sb] - 1

    # Since subband_samples is a float array, it needs to be cast to unsigned integers.
        subband_samples_quantized = subband_samples.astype('uint32')

    # Forming output bitsream and appending it to the output file.
        bitstream_formatting(outmp3file, params,
                             subband_bit_allocation, scfindices,
                             subband_samples_quantized)

        if uniform:

            for ch in range(params.nch):
                for sb in range(N_SUBBANDS):
                    if uniform_bit_allocation[ch, sb] != 0:
                        subband_samples_uniform[ch, sb, :] *= \
                            params_uniform.table.qca[uniform_bit_allocation[ch,
                                sb] - 2]
                        subband_samples_uniform[ch, sb, :] += \
                            params_uniform.table.qcb[uniform_bit_allocation[ch,
                                sb] - 2]
                        subband_samples_uniform[ch, sb, :] *= 1 \
                            << uniform_bit_allocation[ch, sb] - 1

            subband_samples_uniform = \
                subband_samples_uniform.astype('uint32')

            bitstream_formatting(outmp3file[:-4] + '_uniform'
                                 + outmp3file[-4:], params_uniform,
                                 uniform_bit_allocation, scfindices,
                                 subband_samples_uniform)


def newfigure(*args, **kwargs):
    """Create a new figure with golden ratio."""

    xsize = 10
    ysize = xsize * 2 / (1 + np.sqrt(5))
    fig = plt.figure(figsize=(xsize, ysize), dpi=80)

    plottype = kwargs.get('plottype', 'default')
    nsubplots = kwargs.get('nsubplots', 1)

    for nsub in range(nsubplots):
        fig.add_subplot(nsubplots, 1, nsub + 1)

    fig.subplots_adjust(hspace=0.4)

    return fig


def format_axis(ax, *args, **kwargs):
    """Format a figure axis to desired type."""

    plottype = kwargs.get('plottype', 'default')
    plottitle = kwargs.get('title', '')

    if plottype == 'spectrum':
        fs = kwargs.get('fs')
        ax.set_xlim([-fs / 2, fs / 2])
        ticks = np.append(range(0, -int(fs / 2), -5000), range(5000,
                          int(fs / 2 + 1), 5000))
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which='both')
        ax.set_xlabel('Frequency [Hz]')
    elif plottype == 'positivespectrum':

        fs = kwargs.get('fs')
        ax.set_xlim([0, fs / 2])
        ticks = range(0, int(fs / 2 + 1), 5000)
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which='both')
        ax.set_xlabel('Frequency [Hz]')
    elif plottype == 'indices':

        xmin = kwargs.get('xmin', 0)
        xmax = kwargs.get('xmax', 512)
        ax.set_xlim(xmin - 1, xmax + 1)
        ticks = range(int(xmin), int(xmax + 1), int((xmax - xmin) / 16))
        ax.set_xticks(ticks)
        ax.set_title(plottitle)
        ax.grid(True, which='both')

    return ax


def hear_mapping(data, map):
    res = np.zeros(FFT_HALF)
    for i in range(FFT_HALF):
        res[i] = data[map[i]]
    return res


def mask_mapping(data, map):
    res = np.zeros(FFT_HALF)
    for i in range(FFT_HALF):
        res[i] = add_db(data[map[i]])
    return res


def gmask_mapping(data, map):
    res = np.zeros(FFT_HALF)
    for i in range(FFT_HALF):
        res[i] = add_db((data[map[i]], ))
    return res


def masking_function_tonal(X, j, table):
    """Calculate a masking function of a tonal component at index j."""

    masking_tonal = []

    for i in range(table.subsize):
        masking_tonal.append(())
        zi = table.bark[i]
        zj = table.bark[table.map[j]]
        dz = zi - zj
        if dz >= -3 and dz <= 8:
            avtm = -1.525 - 0.275 * zj - 4.5
            if dz >= -3 and dz < -1:
                vf = 17 * (dz + 1) - (0.4 * X[j] + 6)
            elif dz >= -1 and dz < 0:
                vf = dz * (0.4 * X[j] + 6)
            elif dz >= 0 and dz < 1:
                vf = -17 * dz
            else:
                vf = -(dz - 1) * (17 - 0.15 * X[j]) - 17
            masking_tonal[i] += (X[j] + vf + avtm, )

    mask = mask_mapping(masking_tonal, table.map)
    return mask


def get_critical_bands(table):
    """Return critical band index boundaries."""

    cbands = [0]
    for cb in range(table.cbnum):
        cbands.append(table.cbound[cb])
    cbands.append(FFT_HALF)
    return cbands