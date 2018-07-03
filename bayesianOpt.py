
import numpy as np
import tensorflow as tf
import gpflowSlim as gfs

FLOAT_TYPE = gfs.settings.float_type


class BayesianOptimization(object):

    def __init__(self, func, kernel, logger, num_dims, grid_size, mins, maxes,
                 epochs=200, iterations=5000, initial_epochs=10):
        self.func = func
        self.kernel = kernel
        self.logger = logger

        self.num_dims = num_dims
        self.grid_size = grid_size
        self.mins = mins
        self.maxes = maxes

        self.epochs = epochs
        self.iterations = iterations
        self.initial_epochs = initial_epochs

        self.sess = tf.Session()
        self.init_inputs()
        self.init_function()
        self.initialize_run()
        self.init_graph()
        self.init_ei()

    def init_inputs(self):
        self.X_ph       = tf.placeholder(FLOAT_TYPE, shape=[None, self.num_dims], name='x')
        self.y_ph       = tf.placeholder(FLOAT_TYPE, shape=[None, 1], name='y')
        self.X_pred_ph  = tf.placeholder(FLOAT_TYPE, shape=[None, self.num_dims], name='x_pred')
        self.X_query_ph = tf.placeholder(FLOAT_TYPE, shape=[None, self.num_dims], name='x_query')
        self.y_mu_ph    = tf.placeholder(FLOAT_TYPE, shape=[], name='y_mu')
        self.y_std_ph   = tf.placeholder(FLOAT_TYPE, shape=[], name='y_std')
        self.best_y_ph  = tf.placeholder(FLOAT_TYPE, shape=[], name='best_y')

    def init_function(self):
        self.function_query = self.func(self.X_query_ph)

    def init_graph(self):
        self.model = gfs.models.GPR(self.X_ph, self.y_ph, self.kernel)
        self.loss = self.model.objective
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.infer = self.optimizer.minimize(self.loss)

    def init_ei(self):
        mu, sigma2 = self.model.predict_y(self.X_pred_ph)
        mu = self.y_std_ph * mu + self.y_mu_ph
        sigma2 = sigma2 * self.y_std_ph ** 2
        self.ei = self.compute_ei(mu, sigma2)

    def _train(self, epoch):
        self.sess.run(tf.global_variables_initializer())
        for iter in range(self.iterations):
            _, loss = self.sess.run(
                [self.infer, self.loss],
                feed_dict={self.X_ph: self.X, self.y_ph: self.y
            })
            if (iter+1) % 1000 == 0:
                self.logger.info('Epoch %4d --- Iteration [%4d/%4d]: Loss = %5.4f'%(
                    epoch, iter, self.iterations, loss))

    def initialize_run(self):
        self.X = np.random.uniform(size=[self.initial_epochs, self.num_dims]).astype(FLOAT_TYPE)
        true_x = self.unscale_x(self.X)
        self.true_y = np.expand_dims(self.sess.run(self.function_query, feed_dict={self.X_query_ph: true_x}), 1)
        self.best_y = np.min(self.true_y)
        self.update_y()
        self.logger.info('Initialing Y: [{}]'.format(self.true_y.squeeze()))

    def next_candidate_dimscan(self, model):
        candidate_set = np.random.uniform(size=[self.grid_size, self.num_dims]).astype(FLOAT_TYPE)
        ei = self.sess.run(self.ei, feed_dict={
            self.X_ph: self.X,
            self.y_ph: self.y,
            self.X_pred_ph: candidate_set,
            self.y_mu_ph: self.y_mu,
            self.y_std_ph: self.y_std,
            self.best_y_ph: self.best_y
        })
        max_index = np.argmax(ei)
        return candidate_set[max_index, :], ei[max_index]

    def compute_ei(self, mu, sigma2):
        sigma = tf.sqrt(sigma2)
        u = tf.cast((self.best_y_ph - mu) / sigma, tf.float32)
        sigma = tf.cast(sigma, tf.float32)
        dist = tf.distributions.Normal(loc=0., scale=1.)
        ucdf = dist.cdf(u)
        updf = dist.prob(u)
        ei = sigma * (updf + u * ucdf)
        #self.logger.info "EI mean/std", np.mean(ei), np.std(ei)
        return ei

    def update_y(self):
        self.y_mu, self.y_std = np.mean(self.true_y), np.std(self.true_y)
        self.y = self.scale_y(self.true_y)
        self.best_y = np.minimum(np.min(self.true_y), self.best_y)

    def unscale_x(self, x):
        return (self.maxes - self.mins) * x + self.mins

    def scale_x(self, x):
        return (x - self.mins) / (self.maxes - self.mins)

    def scale_y(self, y):
        return (y - self.y_mu) / self.y_std

    def unscale_y(self, y):
        return y * self.y_std + self.y_mu

    def optimize(self, num_iters=100, path=None):

        for i in range(num_iters):
            self.logger.info("Epoch " + str(i+1) + " of " + str(num_iters) + ":")

            self._train(i)
            next_pt, ei_val = self.next_candidate_dimscan(self.model)
            true_next_pt = self.unscale_x(next_pt)
            self.logger.info(
                '[' + ''.join(['%.2f ']*len(true_next_pt)) % tuple(true_next_pt) + ']')

            f_val = self.sess.run(self.function_query, feed_dict={self.X_query_ph: np.expand_dims(true_next_pt, 0)})

            self.X = np.concatenate([self.X, np.expand_dims(next_pt, 0)], axis=0)
            self.true_y = np.concatenate([self.true_y, np.expand_dims(f_val, 0)], axis=0)
            self.update_y()
            self.logger.info("Objective value = %.3f, EI = %.3f, current best = %.3f" % (f_val, ei_val[0], self.best_y))
            if path is not None:
                with open("results/{}.npz".format(path), "wb") as outfile:
                    np.savez(outfile, all_X=self.unscale_x(self.X), all_y=self.true_y)
