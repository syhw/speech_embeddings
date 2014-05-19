"""Learn embeddings that mimic ABX distances.

speech features -> weights matrix + non-linearity -> E(X)
x                  W (+ b)          T.nnet.simoid                     

Loss L for one X:
L = 0
for all X' in examples:
    L += dist_{emb}(X, X') - dist_{ABX}(X, X')
"""

import theano, time, pandas, sys, os, h5features
import numpy as np
from theano import tensor as T
from theano import shared
from collections import OrderedDict

DIM_EMBEDDING = 100  # emb dim


class Embedding:
    def __init__(self, x, ab, n_in, n_out):
        #n_in will be something like 3 * 40: 3 frames of 40 Mel filterbanks
        rng = np.random.RandomState(1234)
        W_values = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (n_in + n_out)),
            high=4 * np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = shared(value=W_values, name='W', borrow=True)
        self.b = shared(value=np.zeros((n_out,),
            dtype=theano.config.floatX), name='b', borrow=True)
        self.n_values = x.shape[0]
        self.x = shared(np.asarray(x, dtype=theano.config.floatX))
        # ab[ind1][ind2] = ab distance of x[ind1] <-> x[ind2]
        self.ab_npy = ab
        self.ab = shared(np.asarray(ab, dtype=theano.config.floatX))
        self.embed_x = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)

    def cost(self):
        dist = shared(np.float32(0.0))
        for ind1 in xrange(self.n_values):
            for ind2 in self.ab_ind_iterator(ind1):
                dist += self.loss_f(ind1, ind2)
        return dist

    def ab_ind_iterator(self, ind):
        """Iterates only on indices for which we have an ab distance w/ ind."""
        return [i for i, x in enumerate(self.ab_npy[:,ind] >= 0) if x]

    def project(self, x):
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b)

    def loss_f(self, ind1, ind2):
        """RMSE between normalized distances in embedding vs ABX spaces. """
        return T.sqrt(((self.embed_x[ind1] - self.embed_x[ind2]).norm(2)  # norm L2
                - self.ab[ind1, ind2])**2)

    def mean_dist(self):
        return T.mean([  # accross the dataset
                T.mean(  # accross these ind1-inds2 pairs
                    [self.loss_f(ind1, ind2)\
                       for ind2 in self.ab_ind_iterator(ind1)])\
                       for ind1 in xrange(self.n_values)])
                       ###for ind1, _ in enumerate(self.x)])
        # TODO directly on the Theano computation graph (w/ self.embed_x)


def consonant_midpoint(features):
    """ Midpoint of the first third of the chunk"""
    n_frames = features.shape[1]
    length_third = int(round(n_frames/3.0))
    midpoint_offset = int(round(length_third/2.0))
    point_i = range(length_third + 1)[midpoint_offset - 1] 
    return features[:,point_i]


def vowel_midpoint(features):
    """ Midpoint of the second third of the chunk"""
    n_frames = features.shape[1]
    length_third = int(round(n_frames/3.0))
    midpoint_offset = int(round(length_third/2.0))
    point_i = range(2*length_third, n_frames + 1)[midpoint_offset - 1] 
    return features[:,point_i]


def get_feature_name_from_h5_fn(fn):
    return os.path.splitext(os.path.basename(fn))[0]                                                       


def get_features_flat(fn, gn, feature_index, key, stat):
    features_dict = h5features.read(fn, gn, key, index=feature_index)[1]
    features = features_dict[features_dict.keys()[0]]
    return stat(features)


def load_speech(fn, gn, feature_index, stat_type, n_features=39):
    if stat_type == "C":
        stat = consonant_midpoint
    elif stat_type == "V":
        stat = vowel_midpoint
    stimuli = feature_index['files']
    n_samples = len(stimuli)
    result = np.empty((n_samples, n_features)) ### TODO
    #result = np.empty((3, n_features))
    for i in range(n_samples): ### TODO 
    #for i in range(3):
        key = stimuli[i]
        result[i, :] = get_features_flat(fn, gn, feature_index, key, stat)
    return result


def load_sim(fn, feature_index, stat_type):
    stimuli = feature_index['files']
    n_samples = len(stimuli)
    result = -np.ones((n_samples,n_samples))
    all_data = pandas.read_csv(fn) 
    if stat_type == "C":
        sim_subset = all_data['cond'].map(lambda x: x == "C")
    elif stat_type == "V":
        sim_subset = all_data['cond'].map(lambda x: x == "V")
    d_cond = all_data[sim_subset]
    d_cond['sim'] = d_cond['sim'] + 1
    d_grouped = d_cond.groupby(['key1', 'key2'])
    d_aggregated = d_grouped['sim'].aggregate(np.mean)
    d_agg_df = pandas.DataFrame(d_aggregated.unstack())
    all_sims = pandas.DataFrame(data=d_agg_df, index=stimuli, columns=stimuli)
    all_sims_m = np.array(all_sims)
    result = (all_sims_m.T + all_sims_m)/2.0
    return result


def train_embedding(speech_fn, sim_fn, learning_rate=0.01, n_epochs=100, dataset='TODO'):
    print '... loading data'
    speech_gn = get_feature_name_from_h5_fn(speech_fn)
    feature_index = h5features.legacy_read_index(speech_fn, speech_gn)
    speech_data = load_speech(speech_fn, speech_gn, feature_index, "C")
    sim_data = load_sim(sim_fn, feature_index, "C")
    speech_train, sim_train = speech_data, sim_data
    # TODO use 3 or 5 or 7 or more (40 mel-filterbank based) frames centered on middle
    print speech_train
    print sim_train
    print '... setting up training'
    emb = Embedding(speech_train, sim_train, n_in=speech_train.shape[1], 
            n_out=DIM_EMBEDDING)
    cost = emb.cost()
    g_W = T.grad(cost=cost, wrt=emb.W)
    g_b = T.grad(cost=cost, wrt=emb.b)
    updates = OrderedDict({emb.W: emb.W - learning_rate * g_W,
        emb.b: emb.b - learning_rate * g_b})
    train_embedding = theano.function(inputs=[], outputs=cost,
            updates=updates, givens={})

    print '... training the model'
    # TODO early-stopping on a validation set
    best_params = None
    best_cost = np.inf
    start_time = time.clock()
    done_looping = False
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        batch_avg_cost = train_embedding()
        print(('epoch %i, training cost %f %%') % (epoch, batch_avg_cost))
        if batch_avg_cost < best_cost:
            best_params = (emb.W, emb.b)
            best_cost = batch_avg_cost
    end_time = time.clock()
    print(('Optimization complete with best training cost of %f %%') %
            (best_cost))
    print "took time (in seconds):", end_time - start_time


if __name__ == '__main__':
  train_embedding(sys.argv[1], sys.argv[2])
