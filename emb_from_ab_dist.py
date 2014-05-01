"""Learn embeddings that mimic ABX distances.

speech features -> weights matrix + non-linearity -> E(X)
x                  W (+ b)          T.nnet.simoid                     

Loss L for one X:
L = 0
for all X' in examples:
    L += dist_{emb}(X, X') - dist_{ABX}(X, X')
"""

import theano, numpy, time
from theano import tensor as T
from theano import shared

DIM_EMBEDDING = 100  # emb dim


class Embedding:
    def __init__(self, x, ab, n_in, n_out):
        #n_in will be something like 3 * 40: 3 frames of 40 Mel filterbanks
        rng = numpy.random.RandomState(1234)
        W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = shared(value=W_values, name='W', borrow=True)
        self.b = shared(value=numpy.zeros((n_out,),
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
        return T.mean([T.mean([self.loss_f(ind1, ind2) for ind2 in self.ab_ind_iterator(ind1)]) for ind1 in xrange(self.x.eval().shape[0])])


def train_embedding(learning_rate=0.01, n_epochs=100, dataset='TODO'):
    datasets = load_data(dataset)  # TODO
    train_set_x, train_set_ab = datasets[0]
    # train_set_x should be (n_samples, n_features)
    # train_set_ab should be a sparse (n_samples, n_samples),
    # with a value train_set_ab[i][j] < 0 when sample i was not matched with j
    x = T.matrix('x')
    ab = T.matrix('ab')
    x = shared(train_set_x)
    ab = shared(train_set_ab)
    emb = Embedding(x, ab, n_in=x.shape[1], n_out=DIM_EMBEDDING)
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
    best_cost = numpy.inf
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

