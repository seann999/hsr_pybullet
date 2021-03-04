from hsr_env import GraspEnv
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, r):
        for k, v in state.items():
            if k == 'depth':
                state[k] = np.uint16(v * 1000.0)
            else:
                state[k] = v

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(r)

    def __len__(self):
        return len(self.states)

    def sample_batch(self, num):
        b = []

        for _ in range(num):
            b.append(self.sample())

        s, a, r = zip(*b)

        s_dict = {k:np.stack([s_i[k] for s_i in s]) for k in s[0].keys()}

        return s_dict, np.stack(a), np.stack(r)

    def sample(self):
        idx = np.random.randint(len(self.states))
        return self.states[idx], self.actions[idx], self.rewards[idx]


replay = ReplayBuffer()
env = GraspEnv(connect=p.GUI)

for _ in range(1000):
    obs = env.reset()

    # plt.imshow(obs['heightmap'])
    # plt.show()

    if len(replay) > 0:
        s, a, r = replay.sample_batch(4)
        print(s['depth'].shape, a.shape, r.shape)

    y, x = np.where(obs['heightmap'] == obs['heightmap'].max())
    y, x = y[0], x[0]
    loc = [y, x]
    # loc = np.random.randint(0, 300, 2)
    angle = np.random.uniform(0, np.pi * 2)
    success = env.step(loc, angle)
    replay.add(obs['state'], [y, x, angle], float(success))

    print(success)
