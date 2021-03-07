from hsr_env import GraspEnv
import pybullet as p
import pfrl
import torch

from train_agent import QFCN


if __name__ == '__main__':
    config = {'depth_noise': True, 'rot_noise': True}
    env = GraspEnv(check_visibility=True, config=config, n_objects=10, connect=p.GUI)
    q_func = QFCN(debug=False)

    # Set the discount factor that discounts future rewards.
    gamma = 1

    # Use epsilon-greedy for exploration
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        0.5, 0.1, 1000, random_action_func=env.action_space.sample)
    optimizer = torch.optim.Adam(q_func.parameters(), eps=3e-4)
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(capacity=10 ** 6)

    gpu = 0

    class MaxAgent:
        def __init__(self):
            pass

        def act(self, obs):
            hmap = obs[0]
            a = hmap.flatten().argmax()

            return a

    baseline = False

    if baseline:
        agent = MaxAgent()
    else:
        agent = pfrl.agents.DQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            gpu=gpu,
        )

        agent.load('result/test04/best')

    print('>>>>>starting eval')
    max_episode_len = 100
    n_episodes = 100


    def do_trials():
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


    if baseline:
        do_trials()
    else:
        with agent.eval_mode():
            do_trials()

    print('Finished.')
