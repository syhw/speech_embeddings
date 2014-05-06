speech_embeddings
=================

Using embedding-based loss functions for phonetics/speech recognition.

## How to get the phonetics embedding?

    python mlf_to_text.py < ~/postdoc/datasets/TIMIT_train_dev_test/train/train.mlf >> timit_train.txt
or 
    python mlf_to_text.py --forcealigned --timitfoldings < ~/postdoc/datasets/TIMIT_train_dev_test/aligned_train.mlf >> timit_train.txt
    python train_word2vec.py timit_train.txt

