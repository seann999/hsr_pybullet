import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from hsr_env import GraspEnvTask2b
import pybullet as p
import pfrl
import argparse
import cv2
import numpy as np

from train_agent import QFCN, phi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='random')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--show-hmap', action='store_true')
    parser.add_argument('--debug-agent', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--n-objects', type=int, default=30)
    args = parser.parse_args()

    config = {'depth_noise': True, 'rot_noise': True, 'action_grasp': True,
              'action_look': True, 'spawn_mode': 'circle'}
    env = GraspEnvTask2b(config=config, n_objects=args.n_objects, connect=p.GUI if args.gui else p.DIRECT)

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
    random_fn = GraspEnvTask2b.random_action_sample_fn(config)

    def do_trials():
        for i in range(1, n_episodes + 1):
            print('resetting')
            obs = env.reset()
            print('reset done')

            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                print('>>>>')
                if args.show_hmap:
                    cv2.imshow('hmap', np.uint8(obs[0] / obs[0].max() * 255))
                    cv2.waitKey(1)

                # Uncomment to watch the behavior in a GUI window
                # env.render()
                if agent_type == 'random':
                    action = random_fn()
                else:
                    action = agent.act(obs)

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
