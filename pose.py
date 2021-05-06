import pybullet_utils.bullet_client as bc
import pybullet as p
import pybulletX as px
import pybullet_data


c_gui = bc.BulletClient(connection_mode=p.GUI)
c_gui.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = c_gui.loadURDF('plane.urdf')
c_gui.changeDynamics(planeId, -1, lateralFriction=0.5)

px_gui = px.Client(client_id=c_gui._client)
robot = px.Robot('hsrb_description/robots/hsrb.urdf', use_fixed_base=True, physics_client=px_gui)

panel = px.gui.RobotControlPanel(robot)

while True:
    panel.update()
    c_gui.stepSimulation()