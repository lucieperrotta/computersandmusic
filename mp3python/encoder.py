import sys
import os.path
import numpy as np
import psychoacoustic as psycho
from common import *
from parameters import *
	


def main(inwavfile, outmp3file, bitrate):
  """Encoder main function."""

  #inwavfile  = "../samples/sinestereo.wav"
  #outmp3file = "../samples/sinestereo.mp3"
  #bitrate = 320
  
  
  # Read WAVE file and set MPEG encoder parameters.
  input_buffer = WavRead(inwavfile)
  params = EncoderParameters(input_buffer.fs, input_buffer.nch, bitrate)
  

  
  # Subband filter calculation from baseband prototype.
  # Very detailed analysis of MP3 subband filtering available at
  # http://cnx.org/content/m32148/latest/?collection=col11121/latest

  # Read baseband filter samples
  baseband_filter = filter_coeffs()
  # Allocate space for 32 subband filters of length 512.
  filterbank = np.zeros((N_SUBBANDS, FRAME_SIZE), dtype='float32')
  # Perform modulation.
  for sb in range(N_SUBBANDS):
    for n in range(FRAME_SIZE):
      filterbank[sb,n] = baseband_filter[n] * np.cos((2 * sb + 1) * (n - 16) * np.pi / 64)

      

  subband_samples = np.zeros((params.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32') 

  # Main loop, executing until all samples have been processed.
  while input_buffer.nprocessed_samples < input_buffer.nsamples:

    # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
    for frm in range(FRAMES_PER_BLOCK):
      samples_read = input_buffer.read_samples(SHIFT_SIZE)

      # If all samples have been read, perform zero padding.
      if samples_read < SHIFT_SIZE:
        for ch in range(params.nch):
          input_buffer.audio[ch].insert(np.zeros(SHIFT_SIZE - samples_read))

      # Filtering = dot product with reversed buffer.
      for ch in range(params.nch):
        subband_samples[ch,:,frm] = np.dot(filterbank, input_buffer.audio[ch].reversed())
                   

    # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
    # Number of bits allocated in subband is either 0 or in range [2,15].
    scfindices = np.zeros((params.nch, N_SUBBANDS), dtype='uint8')
    subband_bit_allocation = np.zeros((params.nch, N_SUBBANDS), dtype='uint8') 
    smr = np.zeros((params.nch, N_SUBBANDS), dtype='float32')

    
    # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
    # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
    # sound pressure levels.
    for ch in range(params.nch):
      scfindices[ch,:] = get_scalefactors(subband_samples[ch,:,:], params.table.scalefactor)
      subband_bit_allocation[ch,:] = psycho.model1(input_buffer.audio[ch].ordered(), params,scfindices)


    # Scaling subband samples with determined scalefactors.
    for ind in range(FRAMES_PER_BLOCK):
      subband_samples[:,:,ind] /= params.table.scalefactor[scfindices]
  

    # Subband samples quantization. Multiplication with coefficients 'a' and adding coefficients 'b' is
    # defined in the ISO standard.
    for ch in range(params.nch):
      for sb in range(N_SUBBANDS):
        if subband_bit_allocation[ch,sb] != 0:
          subband_samples[ch,sb,:] *= params.table.qca[subband_bit_allocation[ch,sb] - 2]
          subband_samples[ch,sb,:] += params.table.qcb[subband_bit_allocation[ch,sb] - 2]
          subband_samples[ch,sb,:] *= 1<<subband_bit_allocation[ch,sb] - 1
  

    # Since subband_samples is a float array, it needs to be cast to unsigned integers.
    subband_samples_quantized = subband_samples.astype('uint32')


    # Forming output bitsream and appending it to the output file.
    bitstream_formatting(outmp3file,
                         params,
                         subband_bit_allocation,
                         scfindices,
                         subband_samples_quantized)
  



if __name__ == "__main__":
  if len(sys.argv) < 3:
    sys.exit('Please provide input WAVE file and desired bitrate.')
  inwavfile = sys.argv[1]
  if len(sys.argv) == 4:
    outmp3file = sys.argv[2]
    bitrate    = int(sys.argv[3])
  else:
    outmp3file = inwavfile[:-3] + 'mp3'
    bitrate    = int(sys.argv[2])

  if os.path.exists(outmp3file):
    sys.exit('Output file already exists.')

  main(inwavfile,outmp3file,bitrate)
