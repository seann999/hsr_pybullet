import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from scipy.spatial.transform import Rotation as R
import os
import trimesh
import random
import pybullet as p
import numpy as np
import ravens.utils.utils as ru
import numba


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
        shadow=0,
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
    # depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    # depth = (2. * znear * zfar) / depth
    depth = zfar * znear / (zfar - (zfar - znear) * zbuffer)
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

    _, segmaps = ru.reconstruct_heightmaps(
        segs, depths, configs, bounds, px_size)

    return heightmaps, colormaps, segmaps#, rgbs, depths, segs


def load_obj(client, mesh_path, collision_path, rand_scale=True, area=[[0.5, 3.0], [-1.5, 1.5], [0.4, 0.6]]):
    name_log = 'log.txt'

    mesh = trimesh.load(mesh_path, force='mesh', process=False)

    # assert len(mesh.split()) == 1 and mesh.is_watertight

    if not os.path.exists(collision_path):
        p.vhacd(mesh_path, collision_path, name_log)

    if rand_scale:
        max_side = max(mesh.extents)
        scale = np.random.uniform(0.06, 0.35) / max_side
        mesh.apply_scale(scale)
    else:
        scale = 1

    scale = [scale, scale, scale]

    viz_shape_id = client.createVisualShape(
        shapeType=client.GEOM_MESH,
        fileName=mesh_path, meshScale=scale)

    # print(mesh.center_mass)
    # print(mesh.mass)
    # print(success, mesh.is_watertight)
    # print(mesh.moment_inertia)

    col_shape_id = client.createCollisionShape(
        shapeType=client.GEOM_MESH,
        fileName=collision_path, meshScale=scale,
    )

    mesh.density = 150
    # print('CENTER', mesh.center_mass, mesh.mass)

    obj_id = client.createMultiBody(
        baseMass=0.1,
        basePosition=(0, 0, 0),
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=viz_shape_id,
        baseOrientation=(0, 0, 0, 1),
        baseInertialFramePosition=np.array(mesh.center_mass),
    )

    return True, obj_id


def spawn_objects(client, ids=None, ycb=True, num_spawn=None):
    if ycb:
        paths = sorted([x for x in os.listdir('assets/ycb') if os.path.isdir('assets/ycb/{}'.format(x))])
    else:
        paths = sorted([x for x in os.listdir('assets/shapenetsem/original') if x.endswith('.obj')])

    obj_ids = []
    area = [[0.5, 3.0], [-1.5, 1.5], [0.4, 0.6]]

    if ids is None:
        num_spawn = np.random.randint(1, 31) if num_spawn is None else num_spawn
        ids = np.random.randint(0, len(paths), num_spawn)

    # index = 0
    for i in ids:
        #x = folders[i]

        while True:
            x = paths[i]#random.choice(paths)

            if ycb:
                path = 'assets/ycb/{}/google_16k/nontextured.stl'.format(x)
                collision_path = 'assets/ycb/{}/google_16k/collision.obj'.format(x)
                success, obj_id = load_obj(client, path, collision_path, rand_scale=False)
            else:
                path = 'assets/shapenetsem/original/{}'.format(x)
                collision_path = 'assets/shapenetsem/collision/{}'.format(x)
                success, obj_id = load_obj(client, path, collision_path)

            # if not success:
            #     print('failed load')
            #     # index += 1
            #     continue

            # for _ in range(10):
            #     valid = True
            #
            #     for other in obj_ids:
            #         pos = (np.random.uniform(area[0][0], area[0][1]), np.random.uniform(area[1][0], area[1][1]),
            #             np.random.uniform(area[2][0], area[2][1]))
            #         client.resetBasePositionAndOrientation(obj_id, pos, R.random().as_quat())
            #
            #         #if len(client.getClosestPoints(obj_id, other, 0)) > 0:
            #         #    valid = False
            #         break
            #
            #     if valid:
            #         break

            # client.changeVisualShape(obj_id, -1, textureUniqueId=-1)
            # client.changeDynamics(obj_id, -1, lateralFriction=0.5)

            obj_ids.append(obj_id)
            # index += 1

            break

    # for _ in range(240 * 5):
    #     client.stepSimulation()

    return obj_ids


# Read about the noise model here: http://www.alexteichman.com/octo/clams/
# Original source code: http://redwood-data.org/indoor/data/simdepth.py
@numba.jit(nopython=True, fastmath=True)
def undistort(x, y, z, model):
    i2 = int((z + 1) / 2)
    i1 = int(i2 - 1)
    a = (z - (i1 * 2.0 + 1.0)) / 2.0
    x = x // 8
    y = y // 6
    f = (1.0 - a) * model[y, x, min(max(i1, 0), 4)] + a * model[y, x, min(i2, 4)]

    if f < 1e-5:
        return 0.0
    else:
        return z / f


@numba.jit(nopython=True, parallel=True, fastmath=True)
def simulate(gt_depth, model, noise_multiplier):
    noisy_depth = np.empty_like(gt_depth)

    H, W = gt_depth.shape
    ymax, xmax = H - 1.0, W - 1.0

    rand_nums = np.random.randn(H, W, 3).astype(np.float32)

    # Parallelize just the outer loop.  This doesn't change the speed
    # noticably but reduces CPU usage compared to two parallel loops

    for j in numba.prange(H):
        for i in range(W):
            y = int(
                min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = undistort(
                    int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth


DISTORT_MODEL = np.load('redwood-depth-dist-model.npy')
DISTORT_MODEL = DISTORT_MODEL.reshape(DISTORT_MODEL.shape[0], -1, 4)

def distort(depth, noise=1.0):
    return simulate(depth, DISTORT_MODEL, noise)


# class RedwoodNoiseModelCPUImpl:
#     model: np.ndarray
#     noise_multiplier: float
#
#     def __attrs_post_init__(self):
#         self.model = self.model.reshape(self.model.shape[0], -1, 4)
#
#     def simulate(self, gt_depth):
#         return _simulate(gt_depth, self.model, self.noise_multiplier)


if __name__ == '__main__':
    from multiprocessing import Pool

    paths = sorted([x for x in os.listdir('shapenetsem/original') if x.endswith('.obj')])

    def create_collision(x):
        path = 'shapenetsem/original/{}'.format(x)
        collision_path = 'shapenetsem/collision/{}'.format(x)

        if os.path.exists(collision_path):
            return

        p.vhacd(path, collision_path, 'log.txt')

    pool = Pool(8)
    result = pool.map(create_collision, paths)

