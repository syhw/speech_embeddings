#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: reformat_ifa.py
# date: Sun May 04 15:41 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""reformat_ifa: convert IFA format

1. convert 44.1kHz .aifc to 16kHz .wav
2. convert annotation to standard textgrid and strip alignment labels

"""

from __future__ import division

import argparse
import os.path as path
import os
import fnmatch
from subprocess import call
from unicodedata import normalize, category
import cPickle as pickle
import wave
import struct

import numpy as np

import spectral
from textgrid import TextGrid, Tier, Interval


def construct_encoder(config_dict):
    d = {}
    d['nfilt'] = config_dict['nfilters']
    d['do_dct'] = config_dict['cepstral']
    d['compression'] = config_dict['compression']
    d['do_deltas'] = config_dict['deltas']
    d['do_deltasdeltas'] = config_dict['deltas']
    d['fs'] = config_dict['samplerate']
    if 'nceps' in config_dict:
        d['nceps'] = config_dict['nceps']
    return spectral.Spectral(**d)


def get_annot_files(annotroot):
    for root, _, files in os.walk(annotroot):
        for fname in fnmatch.filter(files, '*.phoneme'):
            bname = fname.split('_')[0]
            yield bname, path.relpath(path.join(root, fname), annotroot)


def get_aifc_files(audioroot):
    for root, _, files in os.walk(audioroot):
        for fname in fnmatch.filter(files, '*.aifc'):
            bname = fname.split('_')[0]
            yield bname, path.relpath(path.join(root, fname), audioroot)


def convert_mark(mark, sil):
    ascii = ''.join(c for c in normalize('NFKD',
                                         mark.split('__')[0]
                                         .decode('ISO-8859-2'))
                    if category(c) != 'Mn')
    return ascii.replace('=', '').replace('*', sil)


def convert_textgrid(tg_old, sil):
    tg_new = TextGrid(tg_old.start, tg_old.end)
    phontier = Tier('phones', tg_old['PHONEMES'].start, tg_old['PHONEMES'].end,
                    'Interval',
                    [Interval(x.start, x.end,
                              convert_mark(x.mark, sil))
                     for x in tg_old['PHONEMES']])
    wordtier = Tier('words', tg_old['WORDS'].start, tg_old['WORDS'].end,
                    'Interval',
                    [Interval(x.start, x.end,
                              convert_mark(x.mark, sil))
                     for x in tg_old['WORDS']])
    tg_new.append_tier(wordtier)
    tg_new.append_tier(phontier)
    return tg_new


def convert_to_stack(frames, textgrid, spec, wshift_sec, sil):
    phone_tier = textgrid['phones']
    side = frames // 2
    nphones = sum(1 for p in phone_tier if p.mark != sil)
    rep = np.empty((nphones, spec.shape[1] * frames), dtype=np.double)
    nframes = np.empty((nphones, ), dtype=np.int)
    idx = 0
    offset = 0
    for phone in phone_tier:
        if phone.mark == sil:
            continue
        start_frame = int(np.rint(phone.start / wshift_sec) + offset)
        end_frame = int(np.rint(phone.end / wshift_sec) + offset)
        nframes[idx] = end_frame - start_frame
        center_frame = (start_frame + end_frame) // 2
        if center_frame - side < 0:
            # prepad spec
            spec = np.vstack((np.zeros((abs(center_frame - side),
                                        spec.shape[1])),
                              spec))
            offset = abs(center_frame - side)
            start_frame += offset
            end_frame += offset
            center_frame += offset
        if center_frame + side >= spec.shape[0]:
            # postpad spec
            spec = np.vstack((spec,
                              np.zeros((center_frame + side - spec.shape[0]+1,
                                        spec.shape[1]))))
        # s = spec[center_frame - side: center_frame + side + 1, :]
        datum = np.hstack(spec[center_frame - side:
                               center_frame + side + 1, :])
        rep[idx, :] = datum
        idx += 1
    return rep, nframes


def run(ifaroot, output, encoder, sil, frames, phonesfile):
    annotroot = path.join(ifaroot, 'IFACORPUSfm', 'SLcorpus',
                          'Labels', 'sentences')
    audioroot = path.join(ifaroot, 'IFACORPUSfm', 'SLspeech', 'fm')

    wavdir = path.join(output, 'wav')
    textgriddir = path.join(output, 'textgrid')
    jsondir = path.join(output, 'json')
    featdir = path.join(output, 'feat')
    stackdir = path.join(output, 'stacked')
    phonesdir = path.join(output, 'phones')

    try:
        os.makedirs(wavdir)
    except OSError:
        pass
    try:
        os.makedirs(textgriddir)
    except OSError:
        pass
    try:
        os.makedirs(jsondir)
    except OSError:
        pass
    try:
        os.makedirs(phonesdir)
    except OSError:
        pass
    aifc_files = dict(get_aifc_files(audioroot))

    nframes_total = None
    nsec_total = None
    for annot_base, annot_relpath in get_annot_files(annotroot):
        if not annot_base in aifc_files:
            continue
        print 'converting:', annot_base

        # 1. convert textgrid
        with open(path.join(annotroot, annot_relpath), 'r') as fid:
            tg = convert_textgrid(TextGrid.read(fid), sil)
        nsec = np.array([p.end - p.start for p in tg['phones']
                         if p.mark != sil])
        if nsec_total is None:
            nsec_total = nsec
        else:
            nsec_total = np.hstack((nsec_total, nsec))
        textgrid_file = path.join(textgriddir, annot_base[:4],
                                  annot_base + '.TextGrid')
        if not path.exists(path.dirname(textgrid_file)):
            os.makedirs(path.dirname(textgrid_file))
        with open(textgrid_file, 'w') as fid:
            tg.write(fid)

        # 2. write json
        json_file = path.join(jsondir, annot_base[:4],
                              annot_base + '.json')
        if not path.exists(path.dirname(json_file)):
            os.makedirs(path.dirname(json_file))
        with open(json_file, 'w') as fid:
            fid.write(tg.to_json())

        # 3. convert audio
        wav_file = path.join(wavdir, annot_base[:4],
                             annot_base + '.wav')
        if not path.exists(path.dirname(wav_file)):
            os.makedirs(path.dirname(wav_file))
        call(['sox',
              path.join(audioroot, aifc_files[annot_base]),
              wav_file,
              'gain', '-h',
              'rate', str(encoder.config['fs']),
              'gain', '-rh'])

        # 4. convert to mfc
        feat = convert_audio(wav_file, encoder)
        feat_file = path.join(featdir, annot_base[:4],
                              annot_base + '.npy')
        if not path.exists(path.dirname(feat_file)):
            os.makedirs(path.dirname(feat_file))
        np.save(feat_file, feat)

        # 5. make stacked representation
        fs = encoder.config['fs']
        wshift_sec = fs / encoder.config['frate'] / fs
        rep, nframes = convert_to_stack(frames, tg, feat, wshift_sec, sil)
        if not nframes_total is None:
            nframes_total = np.hstack((nframes_total, nframes))
        else:
            nframes_total = nframes

        stack_file = path.join(stackdir, annot_base[:4],
                               annot_base + '.npy')
        if not path.exists(path.dirname(stack_file)):
            os.makedirs(path.dirname(stack_file))
        np.save(stack_file, rep)

        # 6. make phones
        phones = [p.mark for p in tg['phones'] if p.mark != sil]
        phones_file = path.join(phonesdir, annot_base[:4],
                                annot_base + '.pkl')
        if not path.exists(path.dirname(phones_file)):
            os.makedirs(path.dirname(phones_file))
        with open(phones_file, 'wb') as fid:
            pickle.dump(phones, fid, -1)
    print 'Number of frames per phone:'
    print '    N: ', nframes_total.shape[0]
    print '  avg: ', nframes_total.mean()
    print '  min: ', nframes_total.min()
    print '  max: ', nframes_total.max()
    print
    print 'Length of phones:'
    print '    N: ', nsec_total.shape[0]
    print '  avg: ', nsec_total.mean()
    print '  min: ', nsec_total.min()
    print '  max: ', nsec_total.max()
    np.save('nframes_per_phone.npy', nframes_total)
    np.save('lengths.npy', nsec_total)


def convert_audio(wavfile, encoder):
    fid = wave.open(wavfile, 'r')
    _, _, fs, nframes, _, _ = fid.getparams()
    sig = np.array(struct.unpack_from("%dh" % nframes,
                                      fid.readframes(nframes)))
    fid.close()
    return encoder.transform(sig)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='reformat_ifa.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Reformat IFA corpus; reformat annotation to Praat
TextGrid format and json, plus resample and convert audio from .aifc to .wav""",
        epilog="""Example usage on cluster:

$ ./reformat_ifa.py ifa_reformat_config.py

takes the corpus off fhgfs.""")
    parser.add_argument('config', metavar='CONFIG',
                        nargs=1,
                        help='configuration file')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    config_file = args['config'][0]
    if not path.exists(config_file):
        print 'No such file:', config_file
        exit(1)
    config = {}
    execfile(config_file, config)

    ifaroot = config['corpus']['ifaroot']
    if not path.exists(ifaroot):
        print 'No such directory:', ifaroot
        exit(1)
    output = config['corpus']['output']
    phones_file = config['phones_file']
    encoder = construct_encoder(config['features'])
    with open(config['encoder_file'], 'wb') as fid:
        pickle.dump(encoder, fid, -1)
    sil = config['corpus']['sil']
    nframes = config['frames_around_center']

    run(ifaroot, output, encoder, sil, nframes, phones_file)
