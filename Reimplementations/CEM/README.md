# CEM

Written in CEM.py is a simple implementation of the cross entropy method on OpenAI's CartPole-v0 environment. In this algorithm, policies are parameterized by a set of parameters theta. A distribution of possible theta's is maintained. Theta is sampled from this distribution and a policy is created from the parameters. The policy is evaluated, and the distribution from which we sample theta is adjusted so that the we are more likely to get better thetas. A more in depth explanation may be found here: https://gist.github.com/kashif/5dfa12d80402c559e060d567ea352c06 . When run, the algorithm generates the following learning curve:

![alt text](https://github.com/Ashboy64/rl-reimplementations/blob/master/imgs/cem_cartpole.png)

In the above graph, the x-axis represents the iteration of the algorithm whereas the y-axis represents the average reward achieved by the policies sampled in that iteration.
