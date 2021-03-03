from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import os
import pybullet as p
from tqdm import tqdm
import json

from hsr_env import HSREnv
from env_utils import spawn_ycb


def generate(inputs):
    start, end = inputs

    env = HSREnv(connect=p.DIRECT)

    obj_ids = spawn_ycb(env.c_gui, ids=list(range(79)))
    # print(env.c_gui.getDynamicsInfo(obj_ids[0], -1))
    # masses = [env.c_gui.getDynamicsInfo(id, -1) for id in obj_ids]

    total = end - start
    bar = tqdm(total=total)
    bar.set_description('generating')

    print('objects:', obj_ids)

    for ep in range(start, end):
        for id in obj_ids:
            env.c_gui.resetBasePositionAndOrientation(id, (-100, np.random.uniform(-100, 100), -100), (0, 0, 0, 1))
            env.c_gui.changeDynamics(id, -1, mass=0)

        num_objs = np.random.randint(1, 10)
        selected = np.random.permutation(obj_ids)[:num_objs]

        for id in selected:
            pos = (np.random.uniform(0.5, 2.5), np.random.uniform(-1, 1), np.random.uniform(0.4, 0.6))
            env.c_gui.resetBasePositionAndOrientation(id, pos, R.random().as_quat())
            env.c_gui.changeDynamics(id, -1, mass=0.1)

        for _ in range(240 * 5):
            env.c_gui.stepSimulation()

        env.reset_pose()

        for subep in range(5):
            env.move_joints({
                'head_tilt_joint': np.random.uniform(np.pi * -0.25, 0),
                'head_pan_joint': np.random.uniform(np.pi * -0.25, np.pi * 0.25),
            })

            # hmap, cmap, segmap, rgbs, depths, segs = env.get_heightmap(return_seg=True)
            rgb, depth, seg, config = env.get_heightmap(only_render=True)

            d = 'data/{:05d}_{:02d}'.format(ep, subep)

            try:
                os.makedirs(d)
            except FileExistsError:
                pass

            cv2.imwrite(os.path.join(d, 'rgb.png'), rgb[:, :, ::-1])
            cv2.imwrite(os.path.join(d, 'depth.png'), np.uint16(depth * 1000))
            cv2.imwrite(os.path.join(d, 'seg.png'), seg)

            config['rotation'] = config['rotation'].tolist()

            with open(os.path.join(d, 'config.json'), 'w') as f:
                f.write(json.dumps(config))

        bar.update(1)


if __name__ == '__main__':
    generate((0, 1000))
