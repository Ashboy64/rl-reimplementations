import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import math

class Agent(object):
    """Super simple CartPole Q-learning Agent."""
    def __init__(self, ep_max, ep_len, env, num_buckets):
        super(Agent, self).__init__()
        self.num_buckets = num_buckets
        self.ep_max = ep_max
        self.ep_len = ep_len
        self.env = gym.make(env).unwrapped
        self.q_table = np.zeros(num_buckets + (self.env.action_space.n,))

    def bucket(self, state, bucket_len_arr):
        bucket_indice = []
        bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        bounds[1] = [-0.5, 0.5]
        bounds[3] = [-math.radians(50), math.radians(50)]
        for i in range(len(state)):
            if state[i] <= bounds[i][0]:
                bucket_index = 0
            elif state[i] >= bounds[i][1]:
                bucket_index = bucket_len_arr[i] - 1
            else:
                bound_width = bounds[i][1] - bounds[i][0]
                offset = (bucket_len_arr[i]-1)*bounds[i][0]/bound_width
                scaling = (bucket_len_arr[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def choose_action(self, state, t):
        a = np.argmax(self.q_table[state])
        if random.uniform(0,1) < self.get_explore_rate(t):
            a = self.env.action_space.sample()
        return a

    def get_explore_rate(self, t):
        return max(0.01, min(1, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self, t):
        return max(0.1, min(0.5, 1.0 - math.log10((t+1)/25)))

    def update(self, s_0, s, a, r, t):
        best_q = np.amax(self.q_table[s])
        self.q_table[s_0 + (a,)] += self.get_learning_rate(t)*(r + 0.99*(best_q) - self.q_table[s_0 + (a,)])

    def run(self):
        all_ep_r = []
        for ep in range(self.ep_max):
            s_0 = self.bucket(self.env.reset(), (1,1,6,3))
            ep_r = 0
            for t in range(self.ep_len):
                a = self.choose_action(s_0, ep)
                obv, r, done, _ = self.env.step(a)
                s = self.bucket(obv, (1,1,6,3))
                self.update(s_0, s, a, r, ep)
                s_0 = s
                ep_r += r

                if done:
                    break

            all_ep_r.append(ep_r)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
            )
        return all_ep_r

agent = Agent(ep_max = 200, ep_len = 250, env = "CartPole-v0", num_buckets = (1,1,6,3))
all_ep_r = agent.run()
plt.plot(range(len(all_ep_r)), all_ep_r)
plt.show()
