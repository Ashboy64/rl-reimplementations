# Q Learning
Q learning is an extremely simple learning algorithm in which the 'Q value' of each state-action pair is stored or estimated. The Q value of a state-action pair represents the expected reward that will be gained given the action is taken in that state.  

Implemented in the file is a Q-learning algorithm that encodes the Q value for each state-action pair in a lookup table. The agent is applied to OpenAI's CartPole-v0 environment and achieves the following learning curve:

![alt text](https://github.com/Ashboy64/rl-reimplementations/blob/master/imgs/q_learning_cartpole.png)

In the above graph, the x-axis represents the episode of training whereas the y-axis represents the reward achieved in that episode. Learning rates for the actor and critic were annealed so as to decrease the frequency and magnitude of performance drops during training as a consequence of taking too large of an optimization step. Code to find optimized exploration/learning rates and to discretize the state space was adapted from here: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947 
