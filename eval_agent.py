import cv2

from hsr_env import GraspEnv, to_maps
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import pfrl
import torch
import torch.nn as nn
import gym
import numpy

from train_agent import QFCN

env = GraspEnv(connect=p.GUI)
q_func = QFCN()

# Set the discount factor that discounts future rewards.
gamma = 1

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
    0.5, 0.1, 1000, random_action_func=env.action_space.sample)
optimizer = torch.optim.Adam(q_func.parameters(), eps=3e-4)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
# phi = lambda x: x.astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

# def phi(obs):
#     hmap, cmap = to_maps(obs['rgb'], obs['depth'], obs['config'], env.hmap_bounds, noise=True)
#     hmap = np.stack([hmap, hmap, hmap])
#
#     return hmap

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=50,
    update_interval=1,
    target_update_interval=1,
    minibatch_size=1,
    # phi=phi,
    gpu=gpu,
)

#agent.load('result/result_old/best')
agent.load('result/test02/best')

print('>>>>>starting eval')
max_episode_len = 100
n_episodes = 100

with agent.eval_mode():
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len
            if done or reset:
                break

        print('-----')

print('Finished.')
