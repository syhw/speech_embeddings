import os
import os.path as path

# CONFIGURATION FOR REFORMATTING THE IFA CORPUS
corpus = dict(
    # root directory of the raw IFA corpus:
    ifaroot = path.join(os.environ['HOME'], 'data', 'IFAcorpus'),
    # root directory where to write reformatted corpus to:
    output = path.join(os.environ['HOME'], 'data', 'IFA_reformatted'),
    # symbol for silence
    sil = '#')
features = dict(
    cepstral = False,
    nfilters = 13,
    compression = 'cubicroot',
    deltas = False,
    samplerate = 16000)

# filename to output spectral encoder to
encoder_file = 'ifa_encoder.pkl'
# filename to output phones hdf5 file
phones_file = 'ifa_phones.h5'
# number of frames to take around the center of each phone
frames_around_center = 13
