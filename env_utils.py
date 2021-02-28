import os
import trimesh
import random
import pybullet as p
import numpy as np


def spawn_ycb(client):
    folders = sorted([x for x in os.listdir('ycb') if os.path.isdir('ycb/{}'.format(x))])
    obj_ids = []

    for i in range(0, 5):
        # x = folders[i]
        # x = folders[9]
        x = random.choice(folders)

        path = 'ycb/{}/google_16k/textured.obj'.format(x)

        name_in = path
        collision_path = 'ycb/{}/google_16k/collision.obj'.format(x)
        name_log = 'log.txt'

        if not os.path.exists(collision_path):
            p.vhacd(name_in, collision_path, name_log)

        viz_shape_id = client.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=path, meshScale=1)

        mesh = trimesh.load_mesh(path)
        success = mesh.fill_holes()
        # print(mesh.center_mass)
        # print(mesh.mass)
        # print(success, mesh.is_watertight)
        # print(mesh.moment_inertia)

        col_shape_id = client.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_path, meshScale=1,
        )

        obj_id = client.createMultiBody(
            baseMass=0.1,
            basePosition=(np.random.uniform(0.5, 2.5), np.random.uniform(-1, 1), np.random.uniform(0.4, 0.6)),
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform() * np.pi * 2.0]),
            baseInertialFramePosition=mesh.center_mass,

        )

        client.changeDynamics(obj_id, -1, lateralFriction=0.5)

        obj_ids.append(obj_id)

    for _ in range(240 * 5):
        client.stepSimulation()

    return obj_ids