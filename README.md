# Learn Reinforcement Learning

Repository to learn RL.
- python 3.11.4
- pytorch 2.0.1 + cu117
- gymnasium
- RL algorithms

Before running the codes, make sure that running ```pip install -r requirements.txt``` first. 

## [dqn_tutorial](./dqn_tutorial.ipynb)

Jupyter notebook to deal with simple DQN algorithm. 
Deep Q Network (DQN) is a reinforcement learning algorithm that is model-free, value-based and off-policy. 

It is kind of TD and its time difference is calculated as below. 

$$ \delta = Q(s_t, a_t) - (r_t + \gamma * max_a Q_{target}(s_{t+1}, a)) $$

This notebook trained simple DQN network for two tasks - Acrobot-v1 and CartPole-v1. 

|Acrobot-v1|CartPole-v1|
|-|-|
|![](./images/dqn_tutorial_results/dqn_acrobot.gif)|![](./images/dqn_tutorial_results/dqn_cartpole.gif)|
|![](./images/dqn_tutorial_results/dqn_acrobot.png)|![](./images/dqn_tutorial_results/dqn_cartpole.png)|

## [dqn](./dqn.ipynb)

Jupyter notebook to deal with various dqn-based algorithms. 
In this notebook, the agent is trained on various tasks - Pong-v5, Breakout-v5, Enduro-v5, and DemonAttack-v5. 

### Pong
Example:

![](./images/dqn_results/dqn_pong.gif)

For rainbow algorithm, training status is shown in plot below. 
- **Noise**: Rainbow algorithm includes noisy networks, which perturbes weight and bias of networks. We can see that noise decreases as the agent learns Pong. 
- **Loss**: Loss between model prediction and transition samples. 
- **Frames per episode**: Increases in early training stage as the agent starts to get points, and decreases in the end as the opponent rarely get points. 
- **Score**: Converges around 20.0

![](./images/dqn_results/dqn_pong_plot.png)

Plot below shows test scores for various DQN-based algorithms. All of them shows satisfying scores because Pong is easy environment, but Dueling and Rainbow shows the best performance. 

![](./images/dqn_results/dqn_pong_compare.png)

### Enduro
(Todo)

### Breakout
(Todo)

### DemonAttack
(Todo)

## [reinforce_tutorial](./reinforce_tutorial.ipynb)

Jupyter notebook to deal with simple REINFORCE algorithm. 
REINFORCE is a policy-based reinforcement learning algorithm that is model-free and on-policy. 

It is kind of Monte-Carlo and model is updated after each episode ends. 

$$ \theta \leftarrow \theta + \alpha \sum_{t=0}^T G_t \nabla_\theta \ln \pi(A_t|S_t;\pi_\theta) $$

This notebook trained simple policy network for CartPole-V1 

|CartPole-v1|
|-|
|![](./images/reinforce_tutorial_results/reinforce_cartpole.gif)|
|![](./images/reinforce_tutorial_results/reinforce_cartpole.png)|

## Todo

### Value based
- [x] DQN
- [x] PER (Prioritized Experience Replay)
- [x] Double DQN
- [x] Dueling DQN
- [x] Multi-step learning
- [x] Distributional RL (QR-DQN)
- [x] Noisy Networks
- [x] Rainbow

### Policy based
- [x] REINFORCE
- [ ] Actor Critic
- [ ] A2C (Advantage Actor Critic)
- [ ] A3C (Asynchronous Advantage Actor Critic)
- [ ] DDPG (Deep Deterministic Policy Gradient)
- [ ] PPO (Proximal Policy Optimization)
- [ ] SAC (Soft Actor Critic)
