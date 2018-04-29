# PPO 
Proximal Policy Optimization is an on-policy, actor-critic based RL algorithm. It allows for agents to optimize behavior through the usage of a unique surrogate loss function. Read more about PPO here: https://arxiv.org/pdf/1707.06347.pdf 

PPO presents two surrogate loss functions that may be used to optimize an agent's policy. These include a clipped variant and a KL-Penalty variant. Presented in the "PPO_Clip.py" file is a simple implementation of the clipped variant applied to Open Ai Gym's "CartPole-v0" environment.
When run for 200 episodes on this environment, the following learning curve is generated: 

![alt text](https://github.com/Ashboy64/rl-reimplementations/blob/master/imgs/ppo_clip_cartpole.png)

In the above graph, the x-axis represents the episode of training whereas the y-axis represents the reward achieved in that episode. Learning rates for the actor and critic were linearly annealed so as to decrease the frequency and magnitude of performance drops during training as a consequence of taking too large of a gradient step.
