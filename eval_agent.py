import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from hsr_env import GraspEnv
import pybullet as p
import pfrl
import argparse
import cv2
import numpy as np

from train_agent import QFCN, phi
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


def visualize_grasps(env, q_func):
    plt.clf()

    grasp_map = q_func.last_output[0][0]
    segmap = env.segmap
    pts = []

    for obj in env.obj_ids:
        mask = segmap[:, :, 0] == obj

        if mask.sum() == 0:
            continue

        gmap = grasp_map.copy()
        gmap[:, ~mask] = -100

        print(gmap.max())

        zs, ys, xs = np.where(gmap == gmap.max())
        px = np.stack([zs, ys, xs], axis=1)[0]
        pts.append(px)
        print(px)

    plt.imshow(env.cmap)
    pts = np.array(pts)
    print(pts.shape)
    if len(pts) > 0:
        plt.plot(pts[:, 2], pts[:, 1], 'r*')

        for rot, y, x in pts:
            angle = rot * 2 * np.pi / 16.0 + np.pi*0.5
            print(angle, x, y, np.cos(angle))
            x1 = [x - np.cos(angle) * 10, x + np.cos(angle) * 10]
            x2 = [y - np.sin(angle) * 10, y + np.sin(angle) * 10]
            print(x1, x2)
            plt.plot(x1, x2)

    plt.savefig('grasps.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='random')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--show-hmap', action='store_true')
    parser.add_argument('--shapenet', action='store_true')
    parser.add_argument('--idle', type=int, default=0)
    parser.add_argument('--debug-agent', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--shelf', action='store_true')
    parser.add_argument('--n-objects', type=int, default=30)
    args = parser.parse_args()

    config = {'depth_noise': True, 'rot_noise': True, 'action_grasp': True,
              'action_look': True, 'spawn_mode': 'circle'}
    env = GraspEnv(config=config, n_objects=args.n_objects, connect=p.GUI if args.gui else p.DIRECT,
                   ycb=not args.shapenet, check_object_collision=False, random_hand=not args.shelf, shelf=args.shelf)

    class MaxAgent:
        def __init__(self):
            pass

        def act(self, obs):
            hmap = obs[0]
            a = hmap.flatten().argmax()

            return a

    agent_type = args.agent

    if agent_type == 'max':
        agent = MaxAgent()
    elif agent_type == 'random':
        pass
    else:
        q_func = QFCN(debug=args.debug_agent, pretrain=args.pretrain)
        replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

        gpu = 0

        agent = pfrl.agents.DQN(
            q_func,
            None,
            replay_buffer,
            0,
            None,
            gpu=gpu,
            phi=phi
        )

        if args.model is not None:
            agent.load(args.model)

    print('>>>>>starting eval')
    max_episode_len = 30
    n_episodes = 100

    config = {
        'depth_noise': True,
        'rot_noise': True,
        'action_grasp': True,
        'action_look': True,
        'spawn_mode': 'circle',
        'res': 224,
        'rots': 16,
    }
    random_fn = GraspEnv.random_action_sample_fn(config)

    def do_trials():
        for i in range(1, n_episodes + 1):
            print('resetting')
            obs = env.reset()
            print('reset done')

            if args.idle > 0:
                for _ in range(240 * args.idle):
                    env.stepSimulation()
                continue

            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                print('>>>>')
                if args.show_hmap:
                    # cv2.imshow('hmap', np.uint8(obs[0][0] / obs[0][0].max() * 255))
                    # cv2.waitKey(1)
                    plt.clf()
                    plt.imshow(obs[0][0])#env.cmap)
                    plt.colorbar()
                    plt.show()

                # Uncomment to watch the behavior in a GUI window
                # env.render()
                if agent_type == 'random':
                    action = random_fn()
                elif agent_type == 'argmax':
                    action = obs[0][0].flatten().argmax()
                elif agent_type == 'scripted':
                    action = obs[0][0].flatten().argmax()
                    
                    found_obj = False
                    for id in np.unique(env.segmap):
                        if id in env.obj_ids:
                            found_obj = True
                            py, px = np.where(env.segmap[:, :, 0] == id)
                            idx = np.random.randint(len(py))
                            py, px = py[idx], px[idx]
                            action = py * 224 + px
                    if not found_obj:
                        action = 224 * 224 + 112 * 224 + 112
                else:
                    action = agent.act(obs)
                    visualize_grasps(env, q_func)

                print('acting')
                obs, reward, done, _ = env.step(action)
                print('acted')

                R += reward
                t += 1
                reset = t == max_episode_len
                if done or reset:
                    break

            print(R, '-----')


    if agent_type == 'max' or agent_type == 'random':
        do_trials()
    else:
        with agent.eval_mode():
            do_trials()

    print('Finished.')
