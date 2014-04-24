"""Trains a GMM on formant data (e.g. from the Hillenbrand corpus).
"""

import numpy as np
import pylab as pl
from sklearn.mixture import GMM
from sklearn import metrics
from collections import defaultdict


def parse(fname):
    with open(fname) as f:
        d = map(lambda l: l.rstrip('\n').split(), f.readlines())
    header = d[0]
    d = filter(lambda x: not 'NaN' in x, d)
    return header, np.array(d[1:])


def eval_clusters(y_pred, y, X, gmm):
    # 2D distance based 1-to-1 matching between y_pred and y
    sety = set(y)
    mapsety = dict(zip(xrange(len(sety)), sety))
    assert(len(set(y_pred)) == len(sety))
    maps_to = {}
    d_m = np.ndarray((len(sety), len(sety)))
    for i, phone in enumerate(sety):  # i, phone: ahah!
        for ncomponent in xrange(gmm.means_.shape[0]):
            d_m[i,ncomponent] = np.linalg.norm(gmm.means_[ncomponent]
                - np.mean(X[y==phone]))
    for _ in xrange(d_m.shape[0]):
        indices = np.unravel_index(d_m.argmin(), d_m.shape)
        while mapsety[indices[0]] in maps_to or indices[1] in maps_to.values():
            d_m[indices[0],indices[1]] = np.finfo('d').max
            indices = np.unravel_index(d_m.argmin(), d_m.shape)
        maps_to[mapsety[indices[0]]] = indices[1]
        d_m[indices[0],indices[1]] = np.finfo('d').max
    print maps_to
    y_gold = np.array(map(lambda x: maps_to[x], y))
    print "Adjusted rand scores:",
    print metrics.adjusted_rand_score(y_gold, y_pred)
    print "Homogeneity:",
    print metrics.homogeneity_score(y_gold, y_pred)
    print "Completeness:",
    print metrics.completeness_score(y_gold, y_pred)
    print "V-measure:",
    print metrics.v_measure_score(y_gold, y_pred)
    return y_pred, y_gold, maps_to


if __name__ == "__main__":
    h, d = parse('data/formants.dat')
    X = d[:, 3:5].astype(np.float)
    y = d[:, 2]
    sety = set(y)
    print "All the", len(sety), "vowels:", sety
    gmm = GMM(n_components=len(sety))  # default covar='diag'
    gmm.fit(X)
    y_pred, y_gold, maps_to = eval_clusters(gmm.predict(X), y, X, gmm)
    #pl.scatter(X[:,1], X[:,0], s=20, c=y_gold)
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, 2*len(sety)))
    ax = pl.subplot(2, 1, 1)
    for i, phone in enumerate(sety):  # oups, I did it again
        pl.scatter(X[y==phone,1], X[y==phone,0], s=20, 
                c=colors[2*i], label=phone)
    pl.legend(bbox_to_anchor=(0., 1.02, 1., 1.102), loc=8,
                   ncol=len(sety)/2, mode="expand", borderaxespad=0.)
    ax = pl.subplot(2, 1, 2)
    for i, phone in enumerate(set(y_pred)):  # oups, I did it again
        pl.scatter(X[y_pred==phone,1], X[y_pred==phone,0], s=20, 
                c=colors[2*i], label=phone)
    pl.show()
