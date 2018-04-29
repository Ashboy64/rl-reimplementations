import tensorflow as tf
import numpy as np
import gym
import sys
import random
sys.path.append("C:\\Users\\rao_a\\Desktop\\Coding\\AtomProjects\\rl-reimplementations")
print(sys.path)
from Reimplementations.Utils import *
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, env, ep_max, ep_len, gamma, a_lr, c_lr, batch, a_update_step, c_update_step, s_dim, epsilon):
        super(Agent, self).__init__()
        self.ep_max = ep_max
        self.ep_len = ep_len
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.batch = batch
        self.a_update_step = a_update_step
        self.c_update_step = c_update_step
        self.s_dim = s_dim
        self.sess = tf.Session()
        self.env = gym.make(env).unwrapped
        self.epsilon = epsilon

        self.states_placeholder = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.actions_placeholder = tf.placeholder(tf.int32, [None, 1], 'action')
        self.dicounted_rewards_placeholder = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantages_placeholder = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # critic
        self.value = self.build_critic()
        self.advantage = self.dicounted_rewards_placeholder - self.value
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)

        # actor
        pi, pi_params = self.build_policy('pi', trainable=True)
        oldpi, oldpi_params = self.build_policy('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(pi.logp(self.actions_placeholder) - oldpi.logp(self.actions_placeholder))
                surr = ratio * self.advantages_placeholder
            self.aloss = -tf.reduce_mean(surr)

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.states_placeholder: s, self.dicounted_rewards_placeholder: r})

        # update actor
        [self.sess.run(self.atrain_op, {self.states_placeholder: s, self.actions_placeholder: a, self.advantages_placeholder: adv}) for _ in range(self.a_update_step)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.states_placeholder: s, self.dicounted_rewards_placeholder: r}) for _ in range(self.c_update_step)]

        self.a_lr = max(0.0, self.a_lr - 0.000000125)
        self.c_lr = max(0.0, self.c_lr - 0.0000025)

    def build_policy(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            pd = Probability_Distribution(tf.layers.dense(l2, self.env.action_space.n, trainable=trainable))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        # return norm_dist, params
        return pd, params

    def build_critic(self):
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu)
            val = tf.layers.dense(l1, 1)
            return val

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.states_placeholder: s})
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.value, {self.states_placeholder: s})[0, 0]

    def run(self):
        all_ep_r = []
        for ep in range(self.ep_max):
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(self.ep_len):    # in one episode
                # self.env.render()
                a = self.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r

                # update
                if (t+1) % self.batch == 0 or t == self.ep_len-1:
                    v_s_ = self.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.update(bs, ba, br)

                if done:
                    break

            all_ep_r.append(ep_r)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
            )
        return all_ep_r

agent = Agent("CartPole-v0", 200, 200, 0.9, 0.0001, 0.0002, 32, 10, 10, 4, 0.2)
all_ep_r = agent.run()
plt.plot(range(len(all_ep_r)), all_ep_r)
plt.show()
