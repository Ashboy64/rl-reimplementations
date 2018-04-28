import tensorflow as tf
import numpy as np

class Probability_Distribution():
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = one_hot_actions)

    def sample(self):
        #u = tf.random_uniform(shape = tf.shape(self.logits), dtype = tf.float64)
        u = tf.random_uniform(shape = tf.shape(self.logits), dtype = tf.float32)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis = -1, keep_dims = True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis = -1, keep_dims = True)
        p0 = ea0/z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis = -1)

    def logp(self, x):
        return - self.neglogp(x)
