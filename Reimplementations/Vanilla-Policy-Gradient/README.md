# Vanilla Policy Gradient
This is a simple implementation of the Vanilla Policy Gradient algorithm on OpenAi's CartPole-v0 Environment. This implementation based on tensorflow uses a surrogate loss function to optimize the policy. When applied to OpenAi's CartPole-v0, the following learning curve is generated:

![alt text](https://github.com/Ashboy64/rl-reimplementations/blob/master/imgs/vanilla_pg_cartpole.png)

Where the x-axis represents the episode of training and the y-axis represents the reward achieved in that episode.

## Lessons I learned
It was very interesting looking at some of the mathematical foundations for policy gradient methods (http://karpathy.github.io/2016/05/31/rl/, http://rll.berkeley.edu/deeprlcoursesp17/docs/lec2.pdf). However, vanilla policy gradient methods aren't sample efficient and robust, and frequently take too large gradient steps on the policy that crash performance. Comparing the learning curves of this algorithm with others such as PPO and even Q-learning gives some evidence of this. However, when used in conjunction with penalties/constraints as done in PPO and TRPO, very good performance is achieved.
