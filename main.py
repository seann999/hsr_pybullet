import pybullet as p
import pybullet_data
import pybulletX as px
import pybullet_utils.bullet_client as bc

import collections
from pybulletX.robot_interface import IRobot
from pybulletX.utils.loop_thread import LoopThread
from pybulletX.gui.control_panel import ControlPanel, Sliders, Slider
from pybulletX.helper import flatten_nested_dict, to_nested_dict

bc_direct = bc.BulletClient(connection_mode=p.DIRECT)
bc_gui = bc.BulletClient(connection_mode=p.GUI)

bc_direct.setGravity(0, 0, -9.8)
bc_gui.setGravity(0, 0, -9.8)
#p.setPhysicsEngineParameter(fixedTimeStep=1/600)
#p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
#p.setPhysicsEngineParameter(numSolverIterations=200)
#p.setPhysicsEngineParameter(solverResidualThreshold=1e-10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = bc_gui.loadURDF('plane.urdf')
bc_gui.changeDynamics(planeId, -1, lateralFriction=0.5)

#p.loadURDF('hsrb_description/robots/hsrb.urdf')
px_gui = px.Client(client_id=bc_gui._client)
robot = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_gui)
px_direct = px.Client(client_id=bc_direct._client)
direct_robot = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_direct)

#print(robot.num_dofs, robot.num_joints)
#print(robot.get_joint_infos())

joints = robot.get_joint_infos()
names = joints['joint_name']
states = robot.get_joint_states()
# print(states, len(states))

params = []

uppers, lowers, ranges, rest, damp = [], [], [], [], []

for i, name in enumerate(names):
    ll = joints['joint_lower_limit'][i]
    ul = joints['joint_upper_limit'][i]

    lower = min(ll, ul)
    upper = max(ll, ul)

    lowers.append(lower)
    uppers.append(upper)
    ranges.append(upper - lower)
    rest.append(0)
    damp.append(1)

    # if lower == upper:
    #     continue

    #par = p.addUserDebugParameter(name.decode('utf-8'), lower, upper, 0)
    #params.append(par)
# damp[0] = 0.000001
# damp[1] = 0.000001
# damp[2] = 0.000001

class RobotControlWidget:
    def __init__(self, robot):
        self.robot = robot

        self.sliders = collections.defaultdict(list)

        # get states and action_space
        states = self.robot.get_states()
        action_space = self.robot.action_space
        joint_names = robot.get_joint_infos()['joint_name']

        # turn states and action_space to flattend dictionary
        states = flatten_nested_dict(states)
        action_space = flatten_nested_dict(action_space)

        for key, space in action_space.items():
            if key not in states:
                continue
            state = states[key]
            if isinstance(state, collections.abc.Iterable):
                print(state)
                #names = [f"{key}[{i}]" for i in range(len(state))]
                #name = str(joint_names[i])
                names = [str(joint_names[i]) for i in range(len(state))]
                self.sliders[key] = Sliders(names, space.low, space.high, state)
            else:

                self.sliders[key] = Slider(key, space.low[0], space.high[0], state)

    @property
    def value(self):
        actions = {k: s.value for k, s in self.sliders.items()}
        actions = to_nested_dict(actions)
        return actions


class RobotControlPanel(ControlPanel):
    def __init__(self, robot):
        super().__init__()
        assert isinstance(robot, IRobot)
        self.robot = robot
        self._widget = RobotControlWidget(self.robot)

    def update(self):
        self.robot.set_actions(self._widget.value)

#duck = p.loadURDF('duck_vhacd.urdf', (0.5, 0, 0.1), (0, 0, 0, 1))

# panel = RobotControlPanel(robot)
# panel.start()

# params = []
# for x in ['x', 'y', 'z', 'r', 'p', 'y']:
#     par = p.addUserDebugParameter(x, -4, 4, 0.5 if x == 'z' else 0)
#     params.append(par)

#p.setRealTimeSimulation(1)

# while True:
#     p.stepSimulation()

#print(robot.get_link_states())

# cols = [
#     (1, 0, 0, 1),
#     (0, 1, 0, 1),
#     (0, 0, 1, 1),
#     (0, 0, 0, 1),
#     (1, 1, 1, 1),
#     (1, 1, 0, 1),
#     (0, 1, 1, 1),
#     (1, 0, 1, 1),
# ]
#
# for i in range(40):
#     #34
#     p.changeVisualShape(robot.id, i, rgbaColor=cols[i % len(cols)])
bc_gui.changeVisualShape(robot.id, 34, rgbaColor=(0, 1, 0, 1))
bc_gui.changeVisualShape(robot.id, 35, rgbaColor=(1, 0, 0, 1))
bc_gui.changeVisualShape(robot.id, 38, rgbaColor=(1, 0, 1, 1))
bc_gui.changeVisualShape(robot.id, 40, rgbaColor=(0.5, 0, 0.5, 1))
bc_gui.changeVisualShape(robot.id, 44, rgbaColor=(0, 1, 1, 1))
bc_gui.changeVisualShape(robot.id, 46, rgbaColor=(0, 0.5, 0.5, 1))

vs_id = bc_gui.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
marker_id = bc_gui.createMultiBody(basePosition=[0, 0, 0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
#p.changeVisualShape(robot.id, 40, rgbaColor=(0.5, 0, 0.5, 1))

import time
import numpy as np

DISTAL_OPEN = -np.pi * 0.25
DISTAL_CLOSE = 0
PROXIMAL_OPEN = 1
PROXIMAL_CLOSE = -0.1


def open_gripper():
    open = 1
    q = robot.get_states()['joint_position']
    q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
    q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
    print(q)
    robot.set_actions({'joint_position': q})

    forward()

def close_gripper():
    open = 0
    q = robot.get_states()['joint_position']
    q = list(q)
    q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
    q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
    robot.set_actions({'joint_position': q})

    forward()

def set_joint_position(robot, joint_position, max_forces=None, use_joint_effort_limits=True, sim=True, gui=True):
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

    assert len(robot.free_joint_indices) == len(joint_position), (
        f"number of target positions ({len(joint_position)}) should match "
        f"the number of joint indices ({len(robot.free_joint_indices)})"
    )

    if gui:
        client = bc_gui
        robot = robot
    else:
        client = bc_direct
        robot = direct_robot
    # If not provided, the default value of targetVelocities, positionGains
    # and velocityGains are 0., 0.1, 1.0, respectively.
    #p.setJointMotorControl2(robot.id, robot.free_joint_indices[0], p.POSITION_CONTROL, targetPosition=joint_position[0], maxVelocity=1)
    #p.setJointMotorControl2(robot.id, robot.free_joint_indices[1], p.POSITION_CONTROL, targetPosition=joint_position[1], maxVelocity=1)

    if sim:
        for i in range(len(robot.free_joint_indices)):
            client.setJointMotorControl2(robot.id, robot.free_joint_indices[i], p.POSITION_CONTROL,
                                    targetPosition=joint_position[i], maxVelocity=vels[i], force=force[i])
    else:
        for i in range(len(robot.free_joint_indices)):
            client.resetJointState(robot.id, robot.free_joint_indices[i], targetValue=joint_position[i])
    # p.setJointMotorControlArray(
    #     bodyIndex=robot.id,
    #     jointIndices=robot.free_joint_indices[2:],
    #     controlMode=p.POSITION_CONTROL,
    #     targetPositions=joint_position[2:],
    #     targetVelocities=np.zeros_like(joint_position[2:]),  # default = 0.0
    #     #maxVelocities=[0.1, 0.1, 2. , 0.1 ,1.  ,1. , 0.2 ,1.2 ,2.  ,1.5, 1.5 ,1. , 1. , 1. , 1. ],
    #
    #     #positionGains=np.ones_like(joint_position) * 0.1, # default = 0.1
    #     # velocityGains=np.ones_like(joint_position) * 1.0, # default = 1.0
    # )


def forward(steps=240*5):
    prev = robot.get_states()['joint_position']

    for _ in range(steps):
        bc_gui.stepSimulation()
        curr = robot.get_states()['joint_position']

        if np.all(np.abs(curr - prev) < 1e-4):
            break

        prev = curr


def move_ee(pos, orn, open=True, t=5):
    start = time.time()

    #for i, (x, u, l) in enumerate(zip(r, uppers, lowers)):
    #    r[i] = max(min(x, u), l)

    #print('REST:', r)
    #print(lowers)
    #print(uppers)
    orig_q = robot.get_states()['joint_position']

    # lowers[0] = orig_q[0] - 0.1
    # uppers[0] = orig_q[0] + 0.1
    # lowers[1] = orig_q[1] - 0.1
    # uppers[1] = orig_q[1] + 0.1

    bc_gui.resetBasePositionAndOrientation(marker_id, pos, orn)

    success = False
    from scipy.spatial.transform import Rotation as R

    for i in range(1000):
        lowers[0] = orig_q[0] - 0.05 * i
        uppers[0] = orig_q[0] + 0.05 * i
        lowers[1] = orig_q[1] - 0.05 * i
        uppers[1] = orig_q[1] + 0.05 * i

        #r = list(np.array(robot.action_space.sample()['joint_position']))
        r = list(orig_q)
        q = bc_direct.calculateInverseKinematics(direct_robot.id, 34, pos, orn, lowerLimits=lowers, upperLimits=uppers,
                                     jointRanges=ranges, restPoses=r, maxNumIterations=10000, residualThreshold=1e-5)
        q = list(q)
        #q[2] = q[2] % (np.pi*2)
        set_joint_position(robot, list(q), sim=False, gui=False)
        #set_joint_position(robot, list(q), sim=False)

        g = bc_direct.getLinkState(direct_robot.id, 34, computeForwardKinematics=1)

        pos_err = np.linalg.norm(np.array(g[4]) - pos)
        orn_err = (R.from_quat(g[1]) * R.from_quat(orn).inv()).as_euler('xyz')
        print(i, pos_err, orn_err)
        #print(np.array(g[0]), pos)
        #print(np.array(g[1]), orn)

        if pos_err < 0.01 and np.max(np.abs(orn_err)) < 0.01:
            success = True
            #time.sleep(5)
            break
        else:
            sample = list(np.array(robot.action_space.sample()['joint_position']))
            sample[0] %= 10.0
            sample[1] %= 10.0
            #set_joint_position(robot, sample, sim=False)
            set_joint_position(robot, sample, sim=False, gui=False)

    if not success:
        print('FAILED TO FIND IK')
        #print(np.array(g[0]), pos)
        #print(np.array(g[1]), orn)
        #print('>', np.max(np.array(g[0]) - pos), np.max(np.array(g[1]) - orn))

    q = list(q)
    q[-1] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-2] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE
    q[-3] = DISTAL_OPEN if open else DISTAL_CLOSE
    q[-4] = PROXIMAL_OPEN if open else PROXIMAL_CLOSE

    if q[2] > orig_q[2]:
        diff = (q[2] - orig_q[2]) % (2 * np.pi)
        if diff > np.pi:
            q[2] = orig_q[2] - (np.pi*2 - diff)
        else:
            q[2] = orig_q[2] + diff
    else:
        diff = (orig_q[2] - q[2]) % (2 * np.pi)
        if diff > np.pi:
            q[2] = orig_q[2] + (np.pi*2 - diff)
        else:
            q[2] = orig_q[2] - diff

    set_joint_position(robot, orig_q, sim=False)
    set_joint_position(robot, q)
    steps = 240 * t

    forward(steps)

import os
import trimesh
import random

folders = sorted([x for x in os.listdir('ycb') if os.path.isdir('ycb/{}'.format(x))])
obj_ids = []

print(robot.get_joint_infos())

for i in range(0, 5):
    #x = folders[i]
    #x = folders[9]
    x = random.choice(folders)

    path = 'ycb/{}/google_16k/textured.obj'.format(x)

    name_in = path
    collision_path = 'ycb/{}/google_16k/collision.obj'.format(x)
    name_log = 'log.txt'

    if not os.path.exists(collision_path):
        bc_gui.vhacd(name_in, collision_path, name_log)

    viz_shape_id = bc_gui.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        fileName=path, meshScale=1)

    mesh = trimesh.load_mesh(path)
    success = mesh.fill_holes()
    print(mesh.center_mass)
    print(mesh.mass)
    print(success, mesh.is_watertight)
    print(mesh.moment_inertia)

    col_shape_id = bc_gui.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=collision_path, meshScale=1,
    )

    obj_id = bc_gui.createMultiBody(
        baseMass=0.1,
        basePosition=(np.random.uniform(0.5, 1.5), np.random.uniform(-1, 1), np.random.uniform(0.4, 0.6)),
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=viz_shape_id,
        baseOrientation=bc_gui.getQuaternionFromEuler([0, 0, np.random.uniform() * np.pi * 2.0]),
        baseInertialFramePosition=mesh.center_mass,

    )

    bc_gui.changeDynamics(obj_id, -1, lateralFriction=0.5)

    obj_ids.append(obj_id)

for _ in range(240*5):
    bc_gui.stepSimulation()

# for _ in range(10):
#     print('open')
#     open_gripper()
#     time.sleep(1)
#     print('close')
#     close_gripper()
#     time.sleep(1)
#open_gripper()

# open_gripper = True
# r = list(robot.get_states()['joint_position'])
# for _ in range(1000):
#
#     #r = np.array(robot.action_space.sample()['joint_position'])
#     #print('sampled r:', r)
#     #print(r > lowers, r < uppers)
#
#     #set_joint_position(robot, list(r), sim=False)
#     #time.sleep(0.2)
#
#     q = p.calculateInverseKinematics(robot.id, 34, (0, 1, 0.5), (0, 0, 0, 1), lowerLimits=lowers, upperLimits=uppers,
#                                              jointRanges=ranges, restPoses=list(r), maxNumIterations=10000)
#     q = list(q)
#     q[-1] = DISTAL_OPEN if open_gripper else DISTAL_CLOSE
#     q[-2] = PROXIMAL_OPEN if open_gripper else PROXIMAL_CLOSE
#     q[-3] = DISTAL_OPEN if open_gripper else DISTAL_CLOSE
#     q[-4] = PROXIMAL_OPEN if open_gripper else PROXIMAL_CLOSE
#     print('solved IK')
#
#     #robot.set_actions({'joint_position': q})
#     set_joint_position(robot, q, sim=False)
#
#     time.sleep(0.2)

for _ in range(100):
    #for _ in range(1000):
    duck = random.choice(obj_ids)

    duck_pos, _ = bc_gui.getBasePositionAndOrientation(duck)
    down = bc_gui.getQuaternionFromEuler([np.pi, 0, 0])

    q = list(robot.get_states()['joint_position'])
    neutral = [0 for _ in joints]
    neutral[0] = q[0]
    neutral[1] = q[1]
    neutral[2] = q[2]
    set_joint_position(robot, neutral)

    forward()

    move_ee(duck_pos + np.array([0, 0, 0.4]), down, open=True)

    move_ee(np.array([duck_pos[0], duck_pos[1], 0.24]), down, open=True)

    close_gripper()

    forward()

    move_ee(duck_pos + np.array([0, 0, 0.4]), down, open=False)

    move_ee(duck_pos + np.array([0, 0, 0.4]), p.getQuaternionFromEuler([0, np.pi*0.5, 0]), open=False)

    move_ee(duck_pos + np.array([0, 0, 0.4]), p.getQuaternionFromEuler([0, np.pi*0.5, 0]), open=True)

    print('===========')

# while True:
#     x, y, z, rr, rp, ry = [p.readUserDebugParameter(par) for par in params]
#     q = p.getQuaternionFromEuler([rr, rp, ry])
#
#     #print(robot.get_link_state(31))
#     #print(robot.get_joint_index_by_name('wrist_roll_joint'))
#
#     q = p.calculateInverseKinematics(robot.id, 34, [x, y, z], q, lowerLimits=lowers, upperLimits=uppers,
#                                      jointRanges=ranges, restPoses=rest, maxNumIterations=100)
#     # print(len(q))
#     # actions = []
#     # for par in params:
#     #     actions.append(p.readUserDebugParameter(par))
#
#     robot.set_actions({'joint_position': q})
#     #p.stepSimulation()
#     time.sleep(0.1)
print('done')
while True:
    time.sleep(1)

