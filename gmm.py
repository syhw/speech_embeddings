"""Trains a GMM on formant data (e.g. from the Hillenbrand corpus).
"""

import numpy as np
from sklearn.mixture import GMM
from sklearn import metrics
import pylab as pl
from collections import defaultdict



def parse(fname):
    with open(fname) as f:
        d = map(lambda l: l.rstrip('\n').split(), f.readlines())
    header = d[0]
    d = filter(lambda x: not 'NaN' in x, d)
    return header, np.array(d[1:])


def eval_clusters(y_pred, y):
    # maximize the 1-to-1 matching between y_pred and y
    sety = set(y)
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for i in xrange(y_pred.shape[0]):
        counts[y[i]][y_pred[i]] += 1
    maps_to = {}
    for y_, yp_c_d in counts.iteritems():
        max_ = 0
        ind_max = None
        for yp, c in yp_c_d.iteritems():
            if c > max_:
                max_ = c
                ind_max = yp
        maps_to[y_] = ind_max
    y_gold = np.array(map(lambda x: maps_to[x], y))
    print "Adjusted rand scores:",
    print metrics.adjusted_rand_score(y_gold, y_pred)
    print "Homogeneity:",
    print metrics.homogeneity_score(y_gold, y_pred)  
    print "Completeness:",
    print metrics.completeness_score(y_gold, y_pred) 
    print "V-measure:",
    print metrics.v_measure_score(y_gold, y_pred)    


if __name__ == "__main__":
    h, d = parse('formants.dat')
    X = d[:,3:5].astype(np.float)
    y = d[:,2]
    print "All the", len(set(y)), "vowels:", set(y)
    
    gmm = GMM(n_components=len(set(y)))
    gmm.fit(X)
    eval_clusters(gmm.predict(X), y)



