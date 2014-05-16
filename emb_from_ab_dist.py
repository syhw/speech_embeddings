"""Learn embeddings that mimic ABX distances.

speech features -> weights matrix + non-linearity -> E(X)
x                  W (+ b)          T.nnet.simoid                     

Loss L for one X:
L = 0
for all X' in examples:
    L += dist_{emb}(X, X') - dist_{ABX}(X, X')
"""

import theano, time, pandas, sys, os
import numpy as np
from theano import tensor as T
from theano import shared
from h5features.read_features import read_features_index, read_features_simple

DIM_EMBEDDING = 100  # emb dim


class Embedding:
    def __init__(self, x, ab, n_in, n_out):
        #n_in will be something like 3 * 40: 3 frames of 40 Mel filterbanks
        rng = np.random.RandomState(1234)
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = shared(value=W_values, name='W', borrow=True)
        self.b = shared(value=np.zeros((n_out,),
            dtype=theano.config.floatX), name='b', borrow=True)
        self.x = x  # all the examples
        self.ab = ab  # ab[ind1][ind2] = ab distance of x[ind1] <-> x[ind2]
        self.embed_x = T.nnet.sigmoid(T.dot(x, self.W) + self.b)

    def ab_ind_iterator(self, ind):
        """Iterates only on indices for which we have an ab distance w/ ind."""
        return [i for i, x in enumerate(self.ab[:,ind] >= 0) if x]

    def project(self, x):
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b)

    def loss_f(self, ind1, ind2):
        """RMSE between normalized distances in embedding vs ABX spaces. """
        x1 = self.x[ind1]
        x2 = self.x[ind2]
        return T.sqrt(((self.project(x1)-self.project(x2)).norm(2)  # norm L2
                - self.ab[ind1, ind2])**2)

    def mean_dist(self):
        return T.mean([T.mean([self.loss_f(ind1, ind2)\
                       for ind2 in self.ab_ind_iterator(ind1)])\
                       for ind1 in xrange(self.x.eval().shape[0])])

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
    stim = {"fn": key, "onset": None, "offset": None}
    features = read_features_simple(fn, gn, feature_index, stim)
    return stat(features)

def load_speech(fn, gn, feature_index, stat_type, n_features=39):
    if stat_type == "C":
        stat = consonant_midpoint
    elif stat_type == "V":
        stat = vowel_midpoint
    stimuli = feature_index['files']
    n_samples = len(stimuli)
    result = np.empty((n_samples, n_features))
    for i in range(n_samples):
        key = stimuli[i]
        result[i,:] = get_features_flat(fn, gn, feature_index, key, stat)
    return result

def reorder_rows_and_columns(df, names):
    names_copy = names[:]
    for x in df.columns:
        if x not in names_copy:
            names_copy.append(x)
    return dataframe[cols]

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
    feature_index = read_features_index(speech_fn, speech_gn)
    speech_data = load_speech(speech_fn, speech_gn, feature_index, "C")
    sim_data = load_sim(sim_fn, feature_index, "C")
    speech_train, sim_train = speech_data, sim_data
    speech = T.matrix('speech')
    speech = shared(speech_train)
    sim = T.matrix('sim')
    sim = shared(sim_train)
    print '... setting up training'
    emb = Embedding(speech, sim, n_in=speech.shape[1], n_out=DIM_EMBEDDING)
    cost = emb.mean_dist()
    g_W = T.grad(cost=cost, wrt=emb.W)
    g_b = T.grad(cost=cost, wrt=emb.b)
    updates = {emb.W: emb.W - learning_rate * g_W,
               emb.b: emb.b - learning_rate * g_b}
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
