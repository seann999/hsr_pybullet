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
import random
import numpy as np
from multiprocessing import Pool
import json
import env_utils as eu
from scipy.spatial.transform import Rotation as R
import time


def generate(seed, indices, args):
    config = {'depth_noise': True, 'rot_noise': True, 'action_grasp': True,
              'action_look': True, 'spawn_mode': 'circle', 'res': 224, 'rots': 16, }

    env = GraspEnv(config=config, connect=p.GUI if args.gui else p.DIRECT, ycb=False, full_range=True)
    env.set_seed(seed)
    # env = make_batch_env(config, n_envs=8)

    for i in indices:
        t = time.time()

        while True:
            obs = env.reset()

            result_data = {
                'robot_id': env.robot.id,
                'furn_ids': env.furn_ids,
                'placed_obj_ids': env.placed_objects,
                'obj_ids': env.obj_ids,
            }

            if args.place:
                loc_name = random.choice([
                    'tray_left',
                    'tray_right',
                    'container_left',
                    'container_right',
                    'bin_left',
                    'bin_right',
                    'drawer_bottom',
                    'drawer_left',
                ])
                target_loc = env.furn_ids[loc_name]
                small = loc_name == 'container_left'
                tries = 0

                while target_loc not in np.unique(env.segmap[:, :, 0]):
                    env.reset_pose()
                    env.move_joints({
                        'joint_rz': np.random.uniform(-np.pi, np.pi),
                        'head_tilt_joint': np.random.uniform(-1.57, 0),
                        # 'head_pan_joint': np.random.uniform(np.pi * -0.25, np.pi * 0.25),
                    }, sim=False)
                    obs = env.update_obs()
                    tries += 1

                    if tries >= 100:
                        break

                if tries >= 100:
                    continue

                for _ in range(240 * 5):
                    env.stepSimulation()

                valid_locs = np.stack(np.where(env.segmap[:, :, 0] == target_loc)).T

                px, py = 1000, 1000

                while not (0 <= px < 224 and 0 <= py < 224):
                    if np.random.random() < 0.1:
                        px = np.random.randint(0, 224)
                        py = np.random.randint(0, 224)
                    else:
                        py, px = valid_locs[np.random.randint(len(valid_locs))]
                        py += np.random.randint(low=-10, high=10)
                        px += np.random.randint(low=-10, high=10)

                place_x = env.hmap_bounds[0, 0] + px * env.px_size
                place_y = env.hmap_bounds[1, 0] + py * env.px_size
                place_v = env.obs_config['base_frame'].dot([place_x, place_y, 1, 1])[:3]
                id = eu.spawn_objects(env.c_gui, num_spawn=1, ycb=False, max_side_len=0.1 if small else 0.2)[0]
                env.c_gui.resetBasePositionAndOrientation(id, place_v, R.random().as_quat())

                env.c_gui.resetBasePositionAndOrientation(env.marker_id, [place_v[0], place_v[1], 0.4], (0, 0, 0, 1))

                contact_obj = False
                objs_in_loc = list(set([c[2] for c in env.c_gui.getContactPoints(target_loc) if c[2] in env.obj_ids]))

                for _ in range(240*5):
                    env.stepSimulation()
                    v, av = env.c_gui.getBaseVelocity(id)
                    env.c_gui.resetBaseVelocity(id, [0, 0, v[2]], [0, 0, 0])

                    if not contact_obj:
                        contact_obj |= any([len(env.c_gui.getContactPoints(id, obj)) > 0 for obj in objs_in_loc])

                for _ in range(240*5):
                    env.stepSimulation()

                    if not contact_obj:
                        contact_obj |= any([len(env.c_gui.getContactPoints(id, obj)) > 0 for obj in objs_in_loc])

                contact_loc = len(env.c_gui.getContactPoints(id, target_loc)) > 0
                contact_other = len([c[2] for c in env.c_gui.getContactPoints(id) if c[2] not in objs_in_loc and c[2] != target_loc]) > 0

                result_data.update({
                    'place': {
                        'target_loc_name': loc_name,
                        'target_loc_id': target_loc,
                        'loc_px': [int(px), int(py)],
                        'loc_world': place_v.tolist(),
                        'contact_loc': contact_loc,
                        'contact_neighbor': contact_obj,
                        'contact_other': contact_other,
                    }
                })

            break
            # print(contact_loc, contact_obj, contact_other)

            # obs = np.dstack([env.hmap for _ in range(3)])
            # x = np.uint8(obs / obs.max() * 255)
            # x[py, px] = (0, 0, 255)
            # cv2.imshow('hmap', x)
            # cv2.waitKey(0)

        segmap = env.segmap
        # maskmap = np.logical_or.reduce([segmap[:, :, 0] == id for id in env.obj_ids])
        # maskmap = np.logical_or.reduce([segmap[:, :, 0] == id for id in [env.furn_ids[11], env.furn_ids[12]]])

        if args.show_maps:
            # cv2.imshow('maskmap', np.uint8(maskmap / maskmap.max() * 255))
            cv2.imshow('hmap', np.uint8(obs[0] / obs[0].max() * 255))
            cv2.waitKey(1)

        d = os.path.join(args.root, '{:07d}'.format(i))

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

        json.dump(result_data, open(os.path.join(d, 'ids.json'), 'w'))

        print('time:', time.time() - t)

    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--place', action='store_true')
    parser.add_argument('--show-maps', action='store_true')
    parser.add_argument('--root', type=str, default='pretrain_test')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    try:
        os.makedirs(args.root)
    except FileExistsError:
        pass

    pool = Pool(args.workers)
    indices = np.array_split(range(100000), args.workers)
    result = pool.starmap(generate, [(i, idx, args) for i, idx in enumerate(indices)])
