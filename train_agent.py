from hsr_env import GraspEnv
import pybullet as p
import numpy as np
import pfrl
import torch
import torch.nn as nn
from fcn_model import FCN
import logging
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import functools
import traceback


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

    plt.title('input heightmap')
    plt.imshow(x.cpu().numpy()[0, 0])
    plt.plot([px_x], [px_y], '*r')

    plt.subplot(132)
    plt.imshow(f)
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(look_out.detach().cpu().numpy()[0, 0])
    plt.colorbar()

    plt.gcf().set_size_inches(20, 8)
    plt.tight_layout()
    #plt.show()
    plt.savefig('debug.png')

    #input('waiting')


class QFCN(nn.Module):
    def __init__(self, debug=False, pretrain=None):
        super().__init__()

        rots = 16

        self.grasp_model = FCN(rots, fast=False)
        self.look_model = FCN(1, fast=True)
        self.debug = debug
        self.last_output = None

        if pretrain:
            self.grasp_model.load_state_dict(torch.load(pretrain))

    def forward(self, x):
        x, grasping = x
        # print('inference:', x.shape)
        bs = len(x)

        #if grasping > 0:
        place_out = torch.zeros((bs, 18, 224, 224)).cuda()
        place_out[:, 17, 112, 37] = 1
        place_out = place_out.view(bs, -1)
        #else:
        out = self.grasp_model(x)
        look_out = self.look_model(x)

        if self.debug:
            show_viz(x, out, look_out)
        self.last_output = [out.detach().cpu().numpy(), look_out.detach().cpu().numpy()]

        out = torch.cat([out, look_out], 1)

        pad = torch.zeros_like(look_out)
        out = torch.cat([out, pad], 1)

        out = out.reshape(bs, -1)  # N x RHW

        place = (grasping > 0).unsqueeze(1)
        out = torch.where(place, place_out, out)

        return pfrl.action_value.DiscreteActionValue(out)


def args2config(args):
    return {
        'depth_noise': args.depth_noise,
        'rot_noise': args.rot_noise,
        'action_grasp': True,
        'action_look': True,
        'spawn_mode': 'circle',
        'res': 224,
        'rots': 16,
    }


def phi(x):
    # normalize heightmap
    return (x[0] - 0.2) / 0.2, x[1]


def make_env(idx, config):
    env = GraspEnv(connect=p.DIRECT, config=config, check_object_collision=False, random_hand=True)
    env.set_seed(idx)
    return env


def make_batch_env(config, n_envs=64):
    vec_env = pfrl.envs.MultiprocessVectorEnv([
        functools.partial(make_env, idx, config)
        for idx, env in enumerate(range(n_envs))
    ])
    
    return vec_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='result/test00')
    parser.add_argument('--depth-noise', action='store_true')
    parser.add_argument('--rot-noise', action='store_true')
    parser.add_argument('--test-run', action='store_true')
    parser.add_argument('--pretrain', type=str, default=None)
    args = parser.parse_args()

    config = args2config(args)

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    with open(os.path.join(args.outdir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    # eval_env = GraspEnv(connect=p.DIRECT, config=config)
    # eval_env = GraspEnv(check_visibility=True, connect=p.DIRECT)
    env = make_batch_env(config, 48)
    q_func = QFCN(pretrain=args.pretrain)

    gamma = 0.5

    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        1, 0.01, 16000, random_action_func=GraspEnv.random_action_sample_fn(config, False))
    # optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(q_func.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10000, betasteps=160000)
    #replay_buffer = pfrl.replay_buffers.ReplayBuffer(10000, 1)

    gpu = 0

    agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=16 if args.test_run else 4000,
        update_interval=4,
        target_update_interval=1000,
        minibatch_size=8 if args.test_run else 8,
        gpu=gpu,
        phi=phi,
        max_grad_norm=100,
    )

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    pfrl.experiments.train_agent_batch_with_evaluation(
        agent,
        env=env,
        steps=160000,
        log_interval=1000,
        eval_n_steps=None,
        eval_n_episodes=10,
        max_episode_len=10,
        eval_interval=1000,
        outdir=args.outdir,
        save_best_so_far_agent=True,
        # eval_env=eval_env,
    )

    print('Finished.')
