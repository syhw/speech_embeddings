"""Trains a Word2Vec model on a corpus, by considering phones as words.

Usage:
    train_word2vec.py file.txt

"""

import sys
from gensim.models import Word2Vec

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print __doc__
        sys.exit(-1)
    phones = []
    with open(sys.argv[1]) as f:
        phones = map(lambda x: x.rstrip('\n').split(), f.readlines())
    model = Word2Vec(phones, size=100, window=5, min_count=5, workers=1)
    model.save("timit_phones.word2vec")
    print "similarity of 'p' and 'b' phones:", model.similarity('p', 'b')
    print "similarity of 'p' and 't' phones:", model.similarity('p', 't')
    print "similarity of 'p' and 'ay' phones:", model.similarity('p', 'ay')
    print "similarity of 'p' and 'sil' phones:", model.similarity('p', 'sil')
    print "similarity of 'ay' and 'er' phones:", model.similarity('ay', 'er')
    print "similarity of 'iy' and 'ix' phones:", model.similarity('iy', 'ix')
    print "similarity of 'v' and 'f' phones:", model.similarity('v', 'f')
    print "similarity of 't' and 'k' phones:", model.similarity('t', 'k')
    print "similarity of 'g' and 'k' phones:", model.similarity('g', 'k')

