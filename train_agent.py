from hsr_env import GraspEnv, to_maps
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import pfrl
import torch
import torch.nn as nn
import gym
import numpy
from fcn_model import FCN


class QFCN(nn.Module):
    def __init__(self):
        super().__init__()

        rots = 16

        self.model = FCN(rots)

    def forward(self, x):
        bs = len(x)
        out = self.model(x)

        out = torch.stack(out)  # R x N x 1 x H x W
        out = out.squeeze(2)  # R x N x H x W

        out = out.permute(1, 0, 2, 3)  # N x R x H x W

        # out = out.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # f = np.hstack([out[0, i, :, :] for i in range(16)])
        # plt.imshow(f)
        # plt.show()

        out = out.reshape(bs, -1)  # N x RHW

        return pfrl.action_value.DiscreteActionValue(out)

if __name__ == '__main__':
    env = GraspEnv(connect=p.DIRECT)
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
        minibatch_size=32,
        # phi=phi,
        gpu=gpu,
        max_grad_norm=1,
    )

    import logging
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=20000,           # Train the agent for 2000 steps
        eval_n_steps=None,       # We evaluate for episodes, not time
        eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
        train_max_episode_len=200,  # Maximum length of each episode
        eval_interval=100,   # Evaluate the agent after every 1000 steps
        outdir='result',      # Save everything to 'result' directory
        save_best_so_far_agent=True,
    )

    print('Finished.')
