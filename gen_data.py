import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from hsr_env import GraspEnv
import pybullet as p
import os
import argparse
import cv2
import numpy as np
from multiprocessing import Pool


def generate(seed, indices, args):
    config = {'depth_noise': True, 'rot_noise': True, 'action_grasp': True,
              'action_look': True, 'spawn_mode': 'circle', 'res': 224, 'rots': 16, }

    env = GraspEnv(config=config, connect=p.GUI if args.gui else p.DIRECT)
    env.set_seed(seed)
    # env = make_batch_env(config, n_envs=8)

    for i in indices:
        obs = env.reset()

        segmap = env.segmap
        maskmap = np.logical_or.reduce([segmap[:, :, 0] == id for id in env.obj_ids])

        if args.show_maps:
            cv2.imshow('maskmap', np.uint8(maskmap / maskmap.max() * 255))
            cv2.imshow('hmap', np.uint8(obs[0] / obs[0].max() * 255))
            cv2.waitKey(1)

        d = 'pretrain_data/{:05d}'.format(i)

        try:
            os.makedirs(d)
        except FileExistsError:
            pass

        cv2.imwrite(os.path.join(d, 'rgb.png'), env.rgb[:, :, ::-1])
        cv2.imwrite(os.path.join(d, 'depth.png'), np.uint16(env.depth * 1000))
        cv2.imwrite(os.path.join(d, 'seg.png'), env.seg)
        cv2.imwrite(os.path.join(d, 'cmap.png'), env.cmap[:, :, ::-1])
        cv2.imwrite(os.path.join(d, 'hmap.png'), np.uint16(obs[0] * 1000))
        cv2.imwrite(os.path.join(d, 'maskmap.png'), np.uint8(maskmap) * 255)
        cv2.imwrite(os.path.join(d, 'segmap.png'), segmap[:, :, 0])

    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--show-maps', action='store_true')
    args = parser.parse_args()

    pool = Pool(8)
    indices = np.array_split(range(10000), 8)
    result = pool.starmap(generate, [(i, idx, args) for i, idx in enumerate(indices)])