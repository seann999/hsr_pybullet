from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import os
import pybullet as p
from tqdm import tqdm
from multiprocessing import Pool

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

            hmap, cmap, segmap, rgbs, depths, segs = env.get_heightmap(return_seg=True)

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.imshow(depths[0])
            # plt.colorbar()
            # plt.show()

            cv2.imshow('cmap', cmap[0][:, :, ::-1])
            cv2.imshow('hmap', np.uint8(hmap[0] / hmap[0].max()*255))

            maskmap = np.logical_or.reduce([segmap[0] == id for id in obj_ids])
            cv2.imshow('maskmap', np.uint8(maskmap / maskmap.max() * 255))
            cv2.waitKey(1)

            d = 'data/{:05d}_{:02d}'.format(ep, subep)

            try:
                os.makedirs(d)
            except FileExistsError:
                pass

            cv2.imwrite(os.path.join(d, 'cmap.png'), cmap[0][:, :, ::-1])
            cv2.imwrite(os.path.join(d, 'hmap.png'), np.uint16(hmap[0]*1000))
            cv2.imwrite(os.path.join(d, 'segmap.png'), segmap[0])
            cv2.imwrite(os.path.join(d, 'maskmap.png'), np.uint8(maskmap)*255)
            cv2.imwrite(os.path.join(d, 'rgb.png'), rgbs[0][:, :, ::-1])
            cv2.imwrite(os.path.join(d, 'depth.png'), np.uint16(depths[0] * 1000))
            cv2.imwrite(os.path.join(d, 'seg.png'), segs[0])

        bar.update(1)

if __name__ == '__main__':
    #p = Pool(10)
    #result = p.map(generate, [(0, 100), (100, 200)])
    #print(result)
    generate((0, 10000))
