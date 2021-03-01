from scipy.spatial.transform import Rotation as R
import os
import trimesh
import random
import pybullet as p
import numpy as np
import ravens.utils.utils as ru


def render_camera(client, config):
    """Render RGB-D image with specified camera configuration."""

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)

    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    lookat = config['position'] + lookdir
    updir = (rotm @ updir).reshape(-1)

    focal_len = config['intrinsics'][0]
    znear, zfar = config['zrange']
    viewm = p.computeViewMatrix(config['position'], lookat, updir)

    # print(R.from_matrix(np.array(viewm).reshape(4, 4).T[:3, :3]).as_euler('xyz'))

    fovh = (config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = config['image_size'][1] / config['image_size'][0]
    projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = client.getCameraImage(
        width=config['image_size'][1],
        height=config['image_size'][0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (config['image_size'][0], config['image_size'][1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if config['noise']:
        color = np.int32(color)
        color += np.int32(np.random.normal(0, 3, config['image_size']))
        color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (config['image_size'][0], config['image_size'][1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    depth = (2. * znear * zfar) / depth
    if config['noise']:
        depth += np.random.normal(0, 0.003, depth_image_size)

    # Get segmentation image.
    segm = np.uint8(segm).reshape(depth_image_size)

    return color, depth, segm


def get_heightmaps(client, configs, bounds=None, return_seg=False, px_size=0.003125):
    if bounds is None:
        bounds = np.array([[-0.5, 0.5], [-0.5, 0.5], [0, 0.28]])

    rgbs, depths, segs = [], [], []

    for config in configs:
        rgb, depth, seg = render_camera(client, config)

        rgbs.append(rgb)
        depths.append(depth)
        segs.append(seg[:, :, None])

    heightmaps, colormaps = ru.reconstruct_heightmaps(
        rgbs, depths, configs, bounds, px_size)

    if return_seg:
        _, segmaps = ru.reconstruct_heightmaps(
            segs, depths, configs, bounds, px_size)

        return heightmaps, colormaps, segmaps

    return heightmaps, colormaps


def spawn_ycb(client, ids=None):
    folders = sorted([x for x in os.listdir('ycb') if os.path.isdir('ycb/{}'.format(x))])
    obj_ids = []

    if ids is None:
        ids = np.random.randint(0, len(folders), 10)

    for i in ids:
        x = folders[i]

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
            baseOrientation=R.random().as_quat(),
            baseInertialFramePosition=mesh.center_mass,

        )

        client.changeDynamics(obj_id, -1, lateralFriction=0.5)

        obj_ids.append(obj_id)

    for _ in range(240 * 5):
        client.stepSimulation()

    return obj_ids