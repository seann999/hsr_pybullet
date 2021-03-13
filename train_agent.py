from hsr_env import GraspEnv
import pybullet as p
import numpy as np
import pfrl
import torch
import torch.nn as nn
from fcn_model import FCN
import logging
import sys
import matplotlib.pyplot as plt
import argparse
import os
import yaml


def show_viz(x, out, look_out):
    out_np = out.detach().cpu().numpy()
    f = np.vstack([np.hstack([out_np[0, i, :, :] for i in range(x * 4, (x + 1) * 4)]) for x in range(4)])

    plt.clf()
    plt.subplot(131)

    action = out_np[0].flatten().argmax()
    res = 224
    loc_idx = action % (res * res)
    px_y = int(loc_idx / res)
    px_x = int(loc_idx % res)

    plt.imshow(x.cpu().numpy()[0, 0])
    plt.plot([px_x], [px_y], '*r')

    plt.subplot(132)
    plt.imshow(f, vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(look_out.detach().cpu().numpy()[0, 0], vmin=0, vmax=1)
    plt.colorbar()

    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.show()


class QFCN(nn.Module):
    def __init__(self, debug=False):
        super().__init__()

        rots = 16

        self.grasp_model = FCN(rots)
        self.look_model = FCN(1)
        self.debug = debug

    def forward(self, x):
        bs = len(x)
        out = self.grasp_model(x)

        out = torch.stack(out)  # R x N x 1 x H x W
        out = out.squeeze(2)  # R x N x H x W

        out = out.permute(1, 0, 2, 3)  # N x R x H x W

        look_out = self.look_model(x)

        if self.debug:
            show_viz(x, out, look_out)

        out = torch.cat([out, look_out], 1)
        out = out.reshape(bs, -1)  # N x RHW

        return pfrl.action_value.DiscreteActionValue(out)


def args2config(args):
    return {
        'depth_noise': args.depth_noise,
        'rot_noise': args.rot_noise,
        'action_grasp': True,
        'action_look': True,
        'spawn_mode': 'circle'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='result/test00')
    parser.add_argument('--depth-noise', action='store_true')
    parser.add_argument('--rot-noise', action='store_true')
    parser.add_argument('--test-run', action='store_true')
    args = parser.parse_args()

    config = args2config(args)

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    with open(os.path.join(args.outdir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    env = GraspEnv(connect=p.DIRECT, config=config)
    # eval_env = GraspEnv(check_visibility=True, connect=p.DIRECT)
    q_func = QFCN()

    gamma = 0.5

    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        0.5, 0.1, 5000, random_action_func=env.random_action_sample)
    optimizer = torch.optim.Adam(q_func.parameters(), eps=3e-4)
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6, betasteps=5000)

    gpu = 0

    agent = pfrl.agents.DQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=1 if args.test_run else 100,
        update_interval=1,
        target_update_interval=1,
        minibatch_size=1 if args.test_run else 32,
        gpu=gpu,
    )

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=50000,
        eval_n_steps=None,
        eval_n_episodes=20,
        train_max_episode_len=200,
        eval_interval=100,
        outdir=args.outdir,
        save_best_so_far_agent=True,
        # eval_env=eval_env,
    )

    print('Finished.')
