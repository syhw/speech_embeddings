""" Uses the *.wrd files from train.scp (mfc) to build the timit "words"
corpus, and then applied a phonemic dictionary on it to print phonemic output.
"""

TIMIT_SCP = "/Users/gabrielsynnaeve/postdoc/datasets/TIMIT/train/train.scp"

total_corpus = []
with open(TIMIT_SCP) as f:
    for line in f:
        fname = line.rstrip('\n').split('.')[0] + '.wrd'
        with open(fname) as rf:
            sentence = []
            for l in rf:
                sentence.append(l.rstrip('\n').split(' ')[-1])
            total_corpus.append(sentence)
        
import re
words_dict = {}
#with open("/Users/gabrielsynnaeve/postdoc/contextual_segmentation/phonology_dict/words.txt") as f:
with open("cmudict.txt") as f:
    for line in f:
        #word, _, phonemes = line.split('\t')
        word, phonemes = line.split('\t')
        words_dict[word.lower()] = map(lambda phn: re.sub('\d+', '', phn.lower()), phonemes.split())

def wrd_to_phn(w):
    w = w.lower()
    if w in words_dict:
        #return ''.join(words_dict[w])
        return ' '.join(words_dict[w])
    return ''
#print words_dict

for sentence in total_corpus:
    s = ["sil"] + map(wrd_to_phn, sentence) + ["sil"]
    print ' '.join(s)


