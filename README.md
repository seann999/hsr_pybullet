# hsr_pybullet

This code was partly used for a robot competition.

Contains scripts for sim2real segmentation and RL for top-down grasping.

![hsr](https://user-images.githubusercontent.com/6847529/143902941-330e2e2d-0c74-4e11-bc4f-74ba043ab4a9.gif)
![image](https://user-images.githubusercontent.com/6847529/143903022-82b8ebb8-af1c-45d9-ad26-23b79dc89df7.png)
![image](https://user-images.githubusercontent.com/6847529/143903132-b735c645-27e5-43af-b15d-e361bb03958f.png)


https://user-images.githubusercontent.com/6847529/143903143-d3dabdec-d87b-4511-b4d5-457957a059b4.mp4



Todo: better documentation, including data preparation (which includes ShapeNet models)

## (incomplete) data preparation

Use https://github.com/sea-bass/ycb-tools to download YCB models.

Place models/ycb in repository (i.e. hsr_pybullet/ycb/002_master_chef_can...)

`train_agent.py` is an example of RL for grasping

## credits
`tmc_wrs_gazebo`, which contains assets for the room based on that of the competition, was copied from https://github.com/hsr-project/tmc_wrs_gazebo/tree/master/tmc_wrs_gazebo_worlds. 

`hsrb_description`, which contains the URDF of the Toyota HSR robot, was copied from https://github.com/hsr-project/hsrb_description.

`hsrb_meshes`, which contains the meshes referenced by the robot URDF, was copied from https://github.com/hsr-project/hsrb_meshes.

Part of the panoptic segmentation code comes from https://github.com/bowenc0221/panoptic-deeplab.

The depth sensor noise model comes from https://github.com/facebookresearch/habitat-sim.

A subset of 3D models from the ShapeNet dataset are used by this project. Due to restrictions, these models are only downloadable from the official ShapeNet dataset website.
