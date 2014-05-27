Speech Embeddings
=================

Using embedding-based loss functions for phonetics/speech recognition.

## ABX-distance based embeddings:

    emb_from_ab_dist.py

TODO write doc

## "phn2vec" embeddings:

### Phonetic annotations

There is no silver bullet, you need phonetically annotated speech corpora
(e.g. TIMIT or the Buckeye corpus).

### Phonemic annotations

Then you can also work on the phonemic annotations, for that you need to
transform words into phonemes. I did a hack-job using the CMU phonemic dict.:

    python timit_words_to_phonemes.py

You need to have the TIMIT corpus with a `train.scp` leading to `*.xyz` files
having corresponding `*.wrd` files with word-level annotation (look at the
constant at the start of `timit_words_to_phonemes.py`).

### How to train the embedding? (Using word2vec from gensim)

    python mlf_to_text.py < ~/postdoc/datasets/TIMIT_train_dev_test/train/train.mlf >> timit_train_from_phones.txt

or 

    python mlf_to_text.py --forcealigned --timitfoldings < ~/postdoc/datasets/TIMIT_train_dev_test/aligned_train.mlf >> timit_train_from_phones.txt
    python train_word2vec.py timit_train_from_phones.txt

Same for the Buckeye corpus.

Comparing two embeddings is as simple as:

    python train_word2vec.py timit_train_from_phones.txt timit_train_from_words.txt

or 

    python train_word2vec.py timit_train_from_phones.txt buckeye_train_from_phones.txt

### Notes on the phone(me)s annotations:

For the Buckeye corpus, "tq" (glotal stop in "cat") folded to "sil".

For the TIMIT corpus, "dx" (flap in "butter") inexistent in "words" (phonemic
annotation) version.
 

