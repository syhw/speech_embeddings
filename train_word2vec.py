"""Trains a Word2Vec model on a corpus, by considering phones as words.

Usage:
    train_word2vec.py file.txt [file2.txt]

"""

import sys
from gensim.models import Word2Vec
import numpy as np
import pylab as pl
from sklearn import manifold


def print_similarity_matrix(sphns, model, model2=None):
    print "      ",
    for phn1 in sphns:
        print phn1, " ",
    print ""
    m = np.ndarray((len(sphns), len(sphns)), dtype=np.float32)
    for i, phn1 in enumerate(sphns):
        print phn1.ljust(4) + ":",
        for j, phn2 in enumerate(sphns):
            sim = model.similarity(phn1, phn2)
            if model2 != None:
                sim -= model2.similarity(phn1, phn2)
            print "%0.2f" % sim,
            m[i][j] = sim
        print ""
    phn_order = [phn for phn in sphns]
    return phn_order, m


def scattertext(X, phns, title):
    assert X.shape[1] == 2
    pl.scatter(X[:,0], X[:,1], s=0)
    for i, phn in enumerate(phns):
        pl.annotate(phn, (X[i,0], X[i,1]))
    pl.title(title)
    pl.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)
    phones = []
    with open(sys.argv[1]) as f:
        phones = map(lambda x: x.rstrip('\n').split(), f.readlines())
    set_phones = set([phn for line in phones for phn in line])
    print "first argument phones:", set_phones

    if len(sys.argv) > 2:
        phones2 = []
        with open(sys.argv[2]) as f:
            phones2 = map(lambda x: x.rstrip('\n').split(), f.readlines())
        set_phones2 = set([phn for line in phones2 for phn in line])
        print "second argument phones:", set_phones2
        print "!!! we are using only the intersection (for cases of phonetic vs phonemic"
        print "check your usecase and if that's what you want!!!"
        set_phones = set_phones.intersection(set_phones2)
        print "using the phone set", set_phones

    nphones = len(set_phones)
    ndims = 10  # dimension of the embedding
    print nphones, "phones"
    print "Skip-gram model:"
    model = Word2Vec(phones, size=ndims, window=5, min_count=5, workers=1)
    X_sg = np.ndarray((nphones, ndims))
    model.save("timit_phones.word2vec")
    model2 = None
    if len(sys.argv) > 2:
        model2 = Word2Vec(phones2, size=ndims, window=5, min_count=5, workers=1)
        #X_sg2 = np.ndarray((nphones, ndims))
        #for i, phn in enumerate(phns):
        #    X_sg2[i] = model[phn]
   
    phns, matr_sg = print_similarity_matrix(set_phones, model, model2)
    for i, phn in enumerate(phns):
        X_sg[i] = model[phn]

    print "CBOW model:"
    model = Word2Vec(phones, size=ndims, window=5, min_count=5, workers=1, sg=0)
    X_cbow = np.ndarray((nphones, ndims))
    if len(sys.argv) > 2:
        model2 = Word2Vec(phones2, size=ndims, window=5, min_count=5, workers=1, sg=0)
        #X_cbow2 = np.ndarray((nphones, ndims))
        #for i, phn in enumerate(phns):
        #    X_cbow2[i] = model[phn]
    phns, matr_cbow = print_similarity_matrix(set_phones, model, model2)
    for i, phn in enumerate(phns):
        X_cbow[i] = model[phn]
    ### Plots the similarity matrices according to these models (in order)
    ax = pl.subplot(2, 1, 1)
    ax.tick_params(labelsize=8, direction='out')
    pl.imshow(matr_sg, interpolation='nearest', cmap=pl.cm.Blues)
    ax.set_xticklabels(phns)
    ax.set_xticks(np.arange(nphones))
    ax.set_yticklabels(phns)
    ax.set_yticks(np.arange(nphones))
    pl.xticks(rotation=-90)
    ax = pl.subplot(2, 1, 2)
    ax.tick_params(labelsize=8, direction='out')
    pl.imshow(matr_cbow, interpolation='nearest', cmap=pl.cm.Blues)
    ax.set_xticklabels(phns)
    ax.set_xticks(np.arange(nphones))
    ax.set_yticklabels(phns)
    ax.set_yticks(np.arange(nphones))
    pl.xticks(rotation=-90)
    pl.show()

    ### Plots MDS and Isomap of the phones vectors
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    X_mds = clf.fit_transform(X_sg)
    scattertext(X_mds, phns, "MDS Skip-gram")
    X_iso = manifold.Isomap(nphones-1, n_components=2).fit_transform(X_sg)
    scattertext(X_iso, phns, "Isomap Skip-gram")
    X_mds = clf.fit_transform(X_cbow)
    scattertext(X_mds, phns, "MDS CBOW")
    X_iso = manifold.Isomap(nphones-1, n_components=2).fit_transform(X_cbow)
    scattertext(X_iso, phns, "Isomap CBOW")

