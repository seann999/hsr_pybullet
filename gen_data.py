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
import json


def generate(seed, indices, args):
    config = {'depth_noise': True, 'rot_noise': True, 'action_grasp': True,
              'action_look': True, 'spawn_mode': 'circle', 'res': 224, 'rots': 16, }

    env = GraspEnv(config=config, connect=p.GUI if args.gui else p.DIRECT, ycb=False)
    env.set_seed(seed)
    # env = make_batch_env(config, n_envs=8)

    for i in indices:
        obs = env.reset()

        segmap = env.segmap
        # maskmap = np.logical_or.reduce([segmap[:, :, 0] == id for id in env.obj_ids])
        # maskmap = np.logical_or.reduce([segmap[:, :, 0] == id for id in [env.furn_ids[11], env.furn_ids[12]]])

        if args.show_maps:
            # cv2.imshow('maskmap', np.uint8(maskmap / maskmap.max() * 255))
            cv2.imshow('hmap', np.uint8(obs[0] / obs[0].max() * 255))
            cv2.waitKey(1)

        d = os.path.join(args.root, '{:05d}'.format(i))

        try:
            os.makedirs(d)
        except FileExistsError:
            pass

        cv2.imwrite(os.path.join(d, 'rgb.png'), env.rgb[:, :, ::-1])
        cv2.imwrite(os.path.join(d, 'depth.png'), np.uint16(env.depth * 1000))
        cv2.imwrite(os.path.join(d, 'noisy_depth.png'), np.uint16(env.noisy_depth * 1000))
        cv2.imwrite(os.path.join(d, 'seg.png'), env.seg)
        cv2.imwrite(os.path.join(d, 'cmap.png'), env.cmap[:, :, ::-1])
        cv2.imwrite(os.path.join(d, 'hmap.png'), np.uint16(env.hmap * 1000))
        # cv2.imwrite(os.path.join(d, 'maskmap.png'), np.uint8(maskmap) * 255)
        cv2.imwrite(os.path.join(d, 'segmap.png'), segmap[:, :, 0])

        json.dump({
            'robot_id': env.env.robot.id,
            'furn_ids': env.furn_ids,
            'placed_obj_ids': env.placed_objects,
            'obj_ids': env.obj_ids,
        }, open(os.path.join(d, 'ids.json'), 'w'))

    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--show-maps', action='store_true')
    parser.add_argument('--root', type=str, default='pretrain_test')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    try:
        os.makedirs(args.root)
    except FileExistsError:
        pass

    pool = Pool(args.workers)
    indices = np.array_split(range(10000), args.workers)
    result = pool.starmap(generate, [(i, idx, args) for i, idx in enumerate(indices)])
