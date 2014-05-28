#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: vq.py
# date: Fri May 02 12:10 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""vq:

"""

from __future__ import division

import os.path as path
import os
import fnmatch
import cPickle as pickle
import numpy as np
from sklearn.cluster import KMeans
import time


def get_stacked_files(stackroot):
    for root, _, files in os.walk(stackroot):
        for fname in fnmatch.filter(files, '*.npy'):
            yield fname, path.relpath(path.join(root, fname), stackroot)


if __name__ == '__main__':
    datadir = path.join(os.environ['HOME'], 'data', 'IFA_reformatted')
    stackdir = path.join(datadir, 'stacked')
    vqdir = path.join(datadir, 'vq')
    files = list(get_stacked_files(stackdir))
    t0 = time.time()
    print 'loading data...',
    X = None
    for f, relpath in files:
        if X is None:
            X = np.load(path.join(stackdir, relpath))
        else:
            X = np.vstack((X, np.load(path.join(stackdir, relpath))))
    print 'done. Time taken: {0:.3f}s'.format(time.time() - t0)
    print X.shape

    t0 = time.time()
    print 'clustering...',
    clf = KMeans(n_clusters=25, n_init=10, n_jobs=-1, precompute_distances=True)
    X_vq = clf.fit(X)
    print 'done. Time taken: {0:.3f}s'.format(time.time() - t0)
    del X
    with open('clf.pkl', 'wb') as fid:
        pickle.dump(clf, fid, -1)

    t0 = time.time()
    print 'predicting...',
    for f, relpath in files:
        X = np.load(path.join(stackdir, relpath))
        vqfile = path.join(vqdir, f[:4], f)
        if not path.exists(path.dirname(vqfile)):
            os.makedirs(path.dirname(vqfile))
        np.save(vqfile, clf.predict(X))
    print 'done. Time taken: {0:.3f}s'.format(time.time() - t0)
