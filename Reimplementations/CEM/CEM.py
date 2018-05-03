import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

class Policy(object):
    """Policy  to select actions"""
    def __init__(self, theta, env, ep_len):
        self.theta = theta
        self.env = gym.make(env)
        self.ep_len = ep_len
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        self.weights = theta[0 : ob_space.shape[0] * ac_space.n].reshape(ob_space.shape[0], ac_space.n)
        self.bias = theta[ob_space.shape[0] * ac_space.n : None].reshape(1, ac_space.n)

    def choose_action(self, obv):
        activation = np.dot(obv, self.weights) + self.bias
        action = np.argmax(activation)
        return action

    def run_episode(self):
        obv = self.env.reset()
        dc_ep_r = 0 # discounted episode r
        ep_r = 0 # episode reward
        for t in range(self.ep_len):
            ac = self.choose_action(obv)
            obv, r, done, _ = self.env.step(ac)
            dc_ep_r += r * 0.9**t
            ep_r += r
        return [dc_ep_r, ep_r]

env = gym.make("CartPole-v0")
dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
mean_theta = np.zeros(dim_theta)
std_theta = np.ones(dim_theta)
avg_rews = []

for itr in range(50):
    extra_cov = max(1.0 - itr / 10, 0) * 2.0**2
    policies = [Policy(theta, "CartPole-v0", 200) for theta in np.random.multivariate_normal(mean=mean_theta,
                                           cov=np.diag(np.array(std_theta**2) + extra_cov),
                                           size=25)]
    dc_rewards = list(map(lambda policy: policy.run_episode()[0], policies))
    avg_reward = np.mean(list(map(lambda policy: policy.run_episode()[1], policies)))
    avg_rews.append(avg_reward)

    elite_indices = np.array(np.argsort(dc_rewards)[-5:], dtype = int)
    elite_policies = [policies[index] for index in elite_indices]
    elite_thetas = [policy.theta for policy in elite_policies]

    mean_theta = np.mean(elite_thetas, axis=0)
    std_theta = np.std(elite_thetas, axis=0)
    print(
        'Itr: %i' % itr,
        "|Avg_Ep_r: %i" % avg_reward,
    )

plt.plot(range(len(avg_rews)), avg_rews)
plt.show()
