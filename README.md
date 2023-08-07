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
|![](images/dqn_tutorial_results/dqn_acrobot.png)|![](images/dqn_tutorial_results/dqn_cartpole.png)|
|![](images/dqn_tutorial_results/dqn_acrobot.gif)|![](images/dqn_tutorial_results/dqn_cartpole.gif)|

## Todo

### Value based
- [x] DQN
- [ ] PER (Prioritized Experience Replay)
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Multi-step learning
- [ ] Distributional RL (C51)
- [ ] Noisy Networks
- [ ] Rainbow

### Policy based
- [ ] REINFORCE
- [ ] Actor Critic
- [ ] A2C (Advantage Actor Critic)
- [ ] A3C (Asynchronous Advantage Actor Critic)
- [ ] DDPG (Deep Deterministic Policy Gradient)
- [ ] PPO (Proximal Policy Optimization)
- [ ] SAC (Soft Actor Critic)
