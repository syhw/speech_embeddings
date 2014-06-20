"""Trains a Word2Vec model on a corpus, by considering phones as words.

Usage:
    train_word2vec.py file.txt [file2.txt]

"""

import sys
from gensim.models import Word2Vec
import numpy as np
import pylab as pl
from sklearn import manifold
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering

BICLUSTER = True
n_clusters = 6  # number of biclusters if true
ndims = 10  # dimension of the embedding


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

    if BICLUSTER:
        #model = SpectralBiclustering(n_clusters=4, method='log',
        model = SpectralCoclustering(n_clusters=n_clusters,
                                             random_state=0)
        model.fit(m)
        print "INDICES:",
        indices = [model.get_indices(i) for i in xrange(n_clusters)]
        print indices
        tmp = []
        for i in xrange(n_clusters):
            tmp.extend([phn_order[indices[i][0][j]] for j in xrange(len(indices[i][0]))])
        phn_order = tmp
        fit_data = m[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        m = fit_data

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
    output_name = sys.argv[1].split('.')[0]
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
    print nphones, "phones"
    print "Skip-gram model:"
    model = Word2Vec(phones, size=ndims, window=5, min_count=5, workers=1)
    X_sg = np.ndarray((nphones, ndims))
    model.save(output_name + ".word2vec_sg")
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
    model.save(output_name + ".word2vec_cbow")
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
    ax = pl.subplot(1, 2, 1)
    ax.tick_params(labelsize=16, direction='out')
    ax.set_title("skip-gram")
    pl.imshow(matr_sg, interpolation='nearest', cmap=pl.cm.Blues)
    ax.set_xticklabels(phns)
    ax.set_xticks(np.arange(nphones))
    ax.set_yticklabels(phns)
    ax.set_yticks(np.arange(nphones))
    pl.xticks(rotation=-90)
    ax = pl.subplot(1, 2, 2)
    ax.set_title("CBOW")
    ax.tick_params(labelsize=16, direction='out')
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

