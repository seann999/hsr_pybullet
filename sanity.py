from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import cv2

from hsr_env import HSREnv
from env_utils import spawn_ycb
import random
import pybullet as p

env = HSREnv()

obj_ids = spawn_ycb(env.c_gui, ids=list(range(20)))

for _ in range(100):
    for id in obj_ids:
        env.c_gui.resetBasePositionAndOrientation(id, (100, np.random.uniform(-100, 100), -100), (0, 0, 0, 1))

    num_objs = np.random.randint(1, 10)
    selected = np.random.permutation(obj_ids)[:num_objs]

    for id in selected:
        pos = (np.random.uniform(0.5, 2.5), np.random.uniform(-1, 1), np.random.uniform(0.4, 0.6))
        env.c_gui.resetBasePositionAndOrientation(id, pos, R.random().as_quat())

    for _ in range(240 * 5):
        env.c_gui.stepSimulation()

    env.reset_pose()
    env.move_joints({
        'head_tilt_joint': np.random.uniform(np.pi * -0.25, 0),
        'head_pan_joint': np.random.uniform(np.pi * -0.25, np.pi * 0.25),
    })

    hmap, cmap, segmap = env.get_heightmap(return_seg=True)
    cv2.imshow('cmap', cmap[0][:, :, ::-1])
    cv2.imshow('hmap', np.uint8(hmap[0] / hmap[0].max()*255))

    segmask = np.logical_or.reduce([segmap[0] == id for id in obj_ids])
    cv2.imshow('segmap', np.uint8(segmask / segmask.max() * 255))
    cv2.waitKey(1)

