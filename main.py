from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import cv2

from hsr_env import HSREnv
from env_utils import spawn_ycb
import random
import pybullet as p

env = HSREnv()

obj_ids = spawn_ycb(env.c_gui)

for _ in range(100):
    env.reset_pose()

    obj_id = random.choice(obj_ids)

    obj_pos, _ = env.c_gui.getBasePositionAndOrientation(obj_id)
    grasp_loc = [obj_pos[0], obj_pos[1], 0.24]
    angle = np.random.uniform(0, 2*np.pi)

    env.move_joints({
        'head_tilt_joint': np.random.uniform(np.pi * -0.25, 0),
        'head_pan_joint': np.random.uniform(np.pi * -0.25, np.pi * 0.25),
    })

    hmap, cmap, segmap, _, _, _ = env.get_heightmap(return_seg=True)
    cv2.imshow('cmap', cmap[0][:, :, ::-1])
    cv2.imshow('hmap', np.uint8(hmap[0] / hmap[0].max()*255))

    segmask = np.logical_or.reduce([segmap[0] == id for id in obj_ids])
    cv2.imshow('segmap', np.uint8(segmask / segmask.max() * 255))
    cv2.waitKey(1)

    env.grasp_primitive(grasp_loc, angle)

    down = p.getQuaternionFromEuler([np.pi, 0, 0])
    orn = R.from_quat(down) * R.from_euler('xyz', [0, np.pi*0.5, 0])
    orn = orn.as_quat()

    #env.move_ee(obj_pos + np.array([0, 0, 0.4]), orn, open=False)
    #env.move_ee(obj_pos + np.array([0, 0, 0.4]), orn, open=True)
    env.holding_pose()

    for _ in range(240):
        env.stepSimulation()

    env.move_base(0, 0, 0)
    env.open_gripper()

    for _ in range(240*5):
        env.stepSimulation()

print('done')
while True:
    env.stepSimulation()
