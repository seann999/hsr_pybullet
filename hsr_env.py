import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import pybulletX as px

from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import env_utils as eu

DISTAL_OPEN = -np.pi * 0.25
DISTAL_CLOSE = 0
PROXIMAL_OPEN = 1
PROXIMAL_CLOSE = -0.1


CAMERA_CONFIG = [{
    'image_size': (480, 640),
    'intrinsics': (537.4933389299223, 0.0, 319.9746375212718, 0.0, 536.5961755975517, 244.54846607953, 0.0, 0.0, 1.0),
    'position': None,
    'rotation': None,
    # 'lookat': (0, 0, 0),
    'zrange': (0.01, 10.),
    'noise': False
}]


# robot.set_actions({'joint_position': q}) exceeds max velocity, so use this fn
def set_joint_position(client, robot, joint_position, max_forces=None, use_joint_effort_limits=True):
    vels = robot.get_joint_infos()['joint_max_velocity']
    force = robot.get_joint_infos()['joint_max_force']
    robot.torque_control = False

    assert not np.all(np.array(max_forces) == 0), "max_forces can't be all zero"

    limits = robot.joint_effort_limits(robot.free_joint_indices)
    if max_forces is None and np.all(limits == 0):
        # warnings.warn(
        #     "Joint maximum efforts provided by URDF are zeros. "
        #     "Set use_joint_effort_limits to False"
        # )
        use_joint_effort_limits = False

    opts = {}
    if max_forces is None:
        if use_joint_effort_limits:
            # Case 1
            opts["forces"] = limits
        else:
            # Case 2: do nothing
            pass
    else:
        if use_joint_effort_limits:
            # Case 3
            opts["forces"] = np.minimum(max_forces, limits)
        else:
            # Case 4
            opts["forces"] = max_forces

    assert len(robot.free_joint_indices) == len(joint_position) or len(robot.free_joint_indices) - 4 == len(
        joint_position), (
        f"number of target positions ({len(joint_position)}) should match "
        f"the number of joint indices ({len(robot.free_joint_indices)})"
    )

    for i in range(len(joint_position)):
        client.setJointMotorControl2(robot.id, robot.free_joint_indices[i], p.POSITION_CONTROL,
                                     targetPosition=joint_position[i], maxVelocity=vels[i], force=force[i])


class HSREnv:
    def __init__(self, connect=p.GUI):
        c_direct = bc.BulletClient(connection_mode=p.DIRECT)
        c_gui = bc.BulletClient(connection_mode=connect)

        c_direct.setGravity(0, 0, -9.8)
        c_gui.setGravity(0, 0, -9.8)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = c_gui.loadURDF('plane.urdf')
        c_gui.changeDynamics(planeId, -1, lateralFriction=0.5)

        px_gui = px.Client(client_id=c_gui._client)
        self.robot = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_gui)
        px_direct = px.Client(client_id=c_direct._client)
        self.robot_direct = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_direct)

        c_gui.changeVisualShape(self.robot.id, 34, rgbaColor=(0, 1, 0, 1))
        c_gui.changeVisualShape(self.robot.id, 35, rgbaColor=(1, 0, 0, 1))
        c_gui.changeVisualShape(self.robot.id, 38, rgbaColor=(1, 0, 1, 1))
        c_gui.changeVisualShape(self.robot.id, 40, rgbaColor=(0.5, 0, 0.5, 1))
        c_gui.changeVisualShape(self.robot.id, 44, rgbaColor=(0, 1, 1, 1))
        c_gui.changeVisualShape(self.robot.id, 46, rgbaColor=(0, 0.5, 0.5, 1))

        self.uppers, self.lowers, self.ranges, self.rest = self.get_robot_info()
        self.max_vels = self.robot.get_joint_infos()['joint_max_velocity']
        self.max_forces = self.robot.get_joint_infos()['joint_max_force']
        self.c_gui, self.c_direct = c_gui, c_direct

        vs_id = c_gui.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.marker_id = c_gui.createMultiBody(basePosition=[0, 0, 0], baseCollisionShapeIndex=-1,
                                               baseVisualShapeIndex=vs_id)

        #print(self.robot.get_joint_infos())

    def get_robot_info(self):
        joints = self.robot.get_joint_infos()
        names = joints['joint_name']
        print(list(enumerate(names)))
        # print(self.robot.get_joint_infos(range(self.robot.num_joints)))
        # print(self.robot.get_joint_info_by_name('head_rgbd_sensor_gazebo_frame_joint'))

        uppers, lowers, ranges, rest = [], [], [], []

        for i, name in enumerate(names):
            ll = joints['joint_lower_limit'][i]
            ul = joints['joint_upper_limit'][i]

            lower = min(ll, ul)
            upper = max(ll, ul)

            lowers.append(lower)
            uppers.append(upper)
            ranges.append(upper - lower)
            rest.append(0)

        return uppers, lowers, ranges, rest

    def get_heightmap(self, only_render=False, **kwargs):
        m_pos, m_orn = self.c_gui.getBasePositionAndOrientation(self.marker_id)
        self.c_gui.resetBasePositionAndOrientation(self.marker_id, (0, 100, 0), (0, 0, 0, 1))

        state = self.robot.get_link_state_by_name('head_rgbd_sensor_gazebo_frame_joint')

        camera_config = copy.deepcopy(CAMERA_CONFIG)
        camera_config[0]['position'] = list(state.world_link_frame_position)

        orn = list(state.world_link_frame_orientation)
        orn = (R.from_quat(orn) * R.from_euler('YZ', [0.5*np.pi, -0.5*np.pi])).as_quat()

        camera_config[0]['rotation'] = orn

        if only_render:
            rgb, depth, seg = eu.render_camera(self.c_gui, camera_config[0])
            out = rgb, depth, seg, camera_config[0]
        else:
            out = eu.get_heightmaps(self.c_gui, camera_config,
                                             bounds=np.array([[0, 3], [-1.5, 1.5], [0, 0.3]]), px_size=0.01, **kwargs)

        self.c_gui.resetBasePositionAndOrientation(self.marker_id, m_pos, m_orn)

        return out

    def open_gripper(self):
        self.gripper_command(True)

    def close_gripper(self):
        self.gripper_command(False)

    def set_joint_position(self, q, gui):
        if gui:
            client, robot = self.c_gui, self.robot
        else:
            client, robot = self.c_direct, self.robot_direct

        set_joint_position(client, robot, q)

    def move_joints(self, config):
        for k, v in config.items():
            j = self.robot.get_joint_info_by_name(k)
            self.c_gui.setJointMotorControl2(self.robot.id, j.joint_index, p.POSITION_CONTROL,
                                         targetPosition=v, maxVelocity=j.joint_max_velocity, force=j.joint_max_force)

        self.steps()

    def gripper_command(self, open):
        q = self.robot.get_states()['joint_position']
        q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
        q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
        q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        self.set_joint_position(q, True)

        self.steps()

    def reset_joints(self, q, gui):
        if gui:
            client, robot = self.c_gui, self.robot
        else:
            client, robot = self.c_direct, self.robot_direct

        for joint_index, joint_angle in zip(robot.free_joint_indices, q):
            client.resetJointState(robot.id, joint_index, joint_angle)

    def holding_config(self):
        q = [0 for _ in self.robot.get_states()['joint_position']]

        q[5] = np.pi * -0.25
        q[8] = np.pi * 0.5
        q[9] = -np.pi * 0.5

        return q

    def reset_pose(self):
        neutral = self.holding_config()
        # neutral[0] = q[0]
        # neutral[1] = q[1]
        # neutral[2] = q[2]

        self.reset_joints(neutral, True)
        self.set_joint_position(neutral, True)

    def holding_pose(self):
        curr_q = list(self.robot.get_states()['joint_position'])
        q = self.holding_config()

        for i in [0, 1, 2]:
            q[i] = curr_q[i]

        self.set_joint_position(q, True)
        self.steps()

    def steps(self, steps=240 * 30, finger_steps=240):
        prev = self.robot.get_states()['joint_position']

        for t in range(steps):
            self.c_gui.stepSimulation()
            curr = self.robot.get_states()['joint_position']

            eq = np.abs(curr - prev) < 1e-4
            timeout = [t >= steps] * len(eq[:-4]) + [t >= finger_steps] * 4

            if np.all(np.logical_or(eq, timeout)):
                # print('breaking at', t, 'steps')
                break

            prev = curr

    def stepSimulation(self):
        self.c_gui.stepSimulation()

    def move_base(self, x, y, angle):
        q = list(self.robot.get_states()['joint_position'])
        q[0] = x
        q[1] = y
        q[2] = angle

        self.set_joint_position(q[:-4], True)
        self.steps()

    def move_ee(self, pos, orn, open=True, t=10):
        orig_q = list(self.robot.get_states()['joint_position'])

        self.c_gui.resetBasePositionAndOrientation(self.marker_id, pos, orn)
        self.reset_joints(orig_q, False)

        lowers, uppers = list(self.lowers), list(self.uppers)
        lowers[2] = 0  # -np.pi * 0.2
        uppers[2] = 0  # np.pi * 0.2

        success = False

        for i in range(1000):
            q = self.c_direct.calculateInverseKinematics(self.robot_direct.id, 34, pos, orn, lowerLimits=lowers,
                                                         upperLimits=uppers,
                                                         jointRanges=self.ranges, restPoses=orig_q,
                                                         maxNumIterations=10000, residualThreshold=1e-5)
            q = list(q)
            self.reset_joints(q, False)

            gripper_state = self.c_direct.getLinkState(self.robot_direct.id, 34, computeForwardKinematics=1)

            pos_err = np.linalg.norm(np.array(gripper_state[4]) - pos)
            orn_err = (R.from_quat(gripper_state[1]) * R.from_quat(orn).inv()).as_euler('xyz')

            # print(i, pos_err, orn_err)

            if pos_err < 0.01 and np.max(np.abs(orn_err)) < 0.01:
                success = True
                break
            else:
                sample = list(np.array(self.robot.action_space.sample()['joint_position']))
                sample[0] %= 10.0
                sample[1] %= 10.0

                self.reset_joints(sample, False)

        if not success:
            print('FAILED TO FIND IK')

        # q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
        # q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
        # q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
        # q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE

        if q[2] > orig_q[2]:
            diff = (q[2] - orig_q[2]) % (2 * np.pi)
            if diff > np.pi:
                q[2] = orig_q[2] - (np.pi * 2 - diff)
            else:
                q[2] = orig_q[2] + diff
        else:
            diff = (orig_q[2] - q[2]) % (2 * np.pi)
            if diff > np.pi:
                q[2] = orig_q[2] + (np.pi * 2 - diff)
            else:
                q[2] = orig_q[2] - diff

        # set_joint_position(robot, orig_q, sim=False)
        self.set_joint_position(q[:-4], True)

        steps = 240 * t
        self.steps(steps)

    def grasp_primitive(self, pos, angle=0):
        pos = np.array(pos)
        down = p.getQuaternionFromEuler([np.pi, 0, angle])

        self.move_ee(pos + np.array([0, 0, 0.3]), down)
        self.open_gripper()
        self.move_ee(pos, down)
        self.close_gripper()
        self.move_ee(pos + np.array([0, 0, 0.3]), down)
