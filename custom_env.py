import os
import time
import pdb
import pybullet as p
import pybullet_data
#import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
#from gym.spaces import Box
import random
import time

ACTION_RANGE = 1.0
OBJ_RANGE = 0.1
Deg2Rad = 3.141592/180.0

class UR5_Robotiq85():

    def __init__(self):
        
        self.serverMode = p.GUI
        # connect to engine servers
        self.physicsClient = p.connect(self.serverMode)
        self.loadURDF_all()
        self.eefID = 8 # ee_link

        self.home_pose()
        p.stepSimulation()
        self.set_cam()

    def set_cam(self):
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 2.0],
            cameraTargetPosition=[0.6, 0.0 , 0.85],
            cameraUpVector=[0, 0, 1])
        self.projectionMatrix = p.computeProjectionMatrixFOV(
        fov=30.0, aspect=1.0, nearVal=0.1, farVal=3.1)
        self.width, self.height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=224, height=224, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)
    
    def loadURDF_all(self):
        self.UR5UrdfPath = "./urdf/ur5_robotiq85.urdf"
        # add search path for loadURDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # p.setTimeStep(0.1)

        # Load URDF
        # define world
        self.planeID = p.loadURDF("plane.urdf")

        tableStartPos = [0.7, 0.0, 0.8]
        tableStartOrientation = p.getQuaternionFromEuler([0, 0, 90.0*Deg2Rad])
        self.tableID = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # define environment
        self.blockPos = [0.6+0.3*(random.random()-0.5), 0.3*(random.random()-0.5) , 0.85]
        #self.blockPos = [0.65, 0.025, 0.87]
        self.blockOri = p.getQuaternionFromEuler([0, 0, 0])
        
        self.boxId = p.loadURDF(
        "./urdf/objects/block.urdf",
        self.blockPos, self.blockOri,
        flags = p.URDF_USE_INERTIA_FROM_FILE,useFixedBase=True)
        robotStartPos = [0.0, 0.0, 0.0]
        robotStartOrn = p.getQuaternionFromEuler([0,0,0])

        print("----------------------------------------")
        print("Loading robot from {}".format(self.UR5UrdfPath))        
        self.robotID = p.loadURDF(self.UR5UrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,flags=p.URDF_USE_INERTIA_FROM_FILE)
                             # flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                    "robotiq_85_left_knuckle_joint",
                     "robotiq_85_right_knuckle_joint",
                            "robotiq_85_left_inner_knuckle_joint",
                            "robotiq_85_right_inner_knuckle_joint",
                            "robotiq_85_left_finger_tip_joint",
                            "robotiq_85_right_finger_tip_joint",
                            "ee_fixed_joint", "world_arm_joint"]
        self.joints=[]
        for i in range(p.getNumJoints(self.robotID)):
            p.enableJointForceTorqueSensor(self.robotID,i)
            info = p.getJointInfo(self.robotID, i)
            self.joints.append(info[1].decode("utf-8"))
            print(info)

    def step(self, action):
        '''removed reward'''
        self.move(action)
        self.next_state_dict = self.get_state()
        reward=0
        self.done=False
        info={}
        return self.next_state_dict, reward, self.done, info

    def reset(self):
        print('reset')
        p.resetSimulation()
        self.loadURDF_all()
        self.eefID = 8 # ee_link

        self.home_pose()
        p.stepSimulation()
        self.set_cam()
        obs=self.get_state()
        return 

    def home_pose(self):
        init_pose=[0.6, 0.0 , 1.15]
        init_ori=p.getQuaternionFromEuler([0,Deg2Rad*90,0])
        jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    init_pose,
                                                    init_ori)
        for i, name in enumerate(self.controlJoints):
            if i > 6:
                break
            self.joint = self.joints.index(name)
            #print(self.joint)
            #p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
            if i < 6:
                p.resetJointState(self.robotID, self.joint, targetValue=jointPos[i], targetVelocity=0)
       
        p.stepSimulation()
    
    def move(self, action,setTime=0.01):
        '''move to relative position by IK joint position control'''

        currentPose = self.getRobotPose()
            
        stepPos=[]
        stepOri=[]
        for i in range(2):
            stepPos.append(currentPose[i] + 0.005* action[i])
        stepPos.append(currentPose[2] - 0.02* (action[2]+0.9))
        stepOri.append(currentPose[3])
        stepOri.append(0.7011236967207826)
        stepOri.append(-currentPose[3])
        stepOri.append(0.7011236967207826)

        jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    stepPos,
                                                    stepOri)
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            if i==6:
                targetJointPos = jointPos[i]+action[3]*0.0005
            else:
                targetJointPos = jointPos[i]

            p.setJointMotorControl2(self.robotID,
                                        joint.id,
                                        p.POSITION_CONTROL,
                                        targetPosition = targetJointPos,
                                        # targetVelocity = 5,
                                        force = joint.maxForce, 
                                        maxVelocity = joint.maxVelocity)
            
        for _ in range(25):
            p.stepSimulation()
        

    def getRobotPose(self):
        currentPos = p.getLinkState(self.robotID, 7)[4]#4
        currentOri = p.getLinkState(self.robotID, 7)[5]#5
        #currentOri = p.getEulerFromQuaternion(currentOri)
        currentPose = []
        currentPose.extend(currentPos)
        currentPose.extend(currentOri)
        return currentPose 

    def getRobotPoseE(self):
        currentPos = p.getLinkState(self.robotID, 7)[4]#4
        currentOri = p.getLinkState(self.robotID, 7)[5]#5
        currentOri = p.getEulerFromQuaternion(currentOri)
        currentPose = []
        currentPose.extend(currentPos)
        currentPose.extend(currentOri)
        return currentPose              

    def get_state(self):
        self.width, self.height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=224, height=224, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)
        depthImg=np.resize(depthImg,(224,224,1))
        obs=np.concatenate((rgbImg,depthImg),axis=-1)
        return obs.T

if __name__=="__main__":
    env=UR5_Robotiq85()
    time.sleep(3)
    env.reset()
    time.sleep(2)