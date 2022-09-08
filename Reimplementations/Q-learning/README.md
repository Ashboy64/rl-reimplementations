# Q Learning
Q learning is an extremely simple learning algorithm in which the 'Q value' of each state-action pair is stored or estimated. The Q value of a state-action pair represents the expected reward that will be gained given the action is taken in that state.  

Implemented in the file is a Q-learning algorithm that encodes the Q value for each state-action pair in a lookup table. The agent is applied to OpenAI's CartPole-v0 environment and achieves the following learning curve:

![alt text](https://github.com/Ashboy64/rl-reimplementations/blob/master/imgs/q_learning_cartpole.png)

In the above graph, the x-axis represents the episode of training whereas the y-axis represents the reward achieved in that episode. Learning rates were annealed so as to decrease the frequency and magnitude of performance drops during training as a consequence of taking too large of an optimization step. Code to find optimized exploration/learning rates and to discretize the state space was adapted from here: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947 

## Lessons I learned
This project really taught me the importance of an effective exploration strategy. First I tried simply used an epsilon-greedy policy, with 10% of the time picking a random action instead of taking the recommended action by the policy. This really didn't seem to be getting anywhere; on average the agent achieved a reward of 50 points, with some 'flukes' occuring of around 100 points. Then I tried to linearly anneal the exploration rate, but even this was giving me some trouble. After researching a little online, I saw that annealing the exploration and learning rates in a logarithmic fashion seemed to produce better results. After adapting the schedules to my implementation, vast increases in performance were observed.
