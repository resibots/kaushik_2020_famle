from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
base_path = str(Path(this_file_path))
import pybullet
import pybullet_data
import time 
import numpy as np
import math
import gym

def add_obstacles(env, obstacles):
    for i, obstacle in enumerate(obstacles):
        # colBoxId = env.p.createCollisionShape(env.p.GEOM_BOX,halfExtents=[(obstacle[1]-obstacle[0])/2.0, (obstacle[3]-obstacle[2])/2.0, 0.04],physicsClientId=env.physicsClient)
        # env.p.createMultiBody(baseMass=30, baseCollisionShapeIndex = colBoxId, basePosition = [(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.04],physicsClientId=env.physicsClient)
        if i > len(obstacles)-5: #Box obstacles
            visBoxId = env.p.createVisualShape(env.p.GEOM_BOX,halfExtents=[(obstacle[1]-obstacle[0])/2.0+0.005, (obstacle[3]-obstacle[2])/2.0+0.005, 0.04+0.005],rgbaColor=[0.1,0.7,0.6, 1.0], specularColor=[0.3, 0.5, 0.5, 1.0],physicsClientId=env.physicsClient)
            env.p.createMultiBody(baseMass=0, baseInertialFramePosition=[0,0,0], baseVisualShapeIndex = visBoxId, useMaximalCoordinates=True, basePosition = [(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.04],physicsClientId=env.physicsClient)

class Reacher_env(gym.Env):
    "Reacher environment"
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60}
    
    def __init__(self, controlStep=0.1, simStep = 0.004, visualizationSpeed = 10.0, goal=[-1.,1.],\
                boturdf= base_path + "/assets/urdf/reacher/reacher.urdf", \
                floorurdf= base_path + "/assets/urdf/reacher/plane.urdf", goal_sampling=False,
                mismatches = np.array([1.,1.,1.,1.,1.,0.,0.,0.,0.,0.])):
        assert controlStep > simStep and int(controlStep/simStep) >=2, "Control step should be at least twce of simStep"
        assert len(mismatches) == 10, "5 dims for action multiplier, 5 dims for joint position noise = 10D list"
        self.p = pybullet
        self.action_space = gym.spaces.Box(low=-np.ones(5), high=np.ones(5), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.__vspeed = visualizationSpeed 
        self.__simStep = simStep
        self.__controlStep = controlStep
        self.physicsClient = object()
        self.boturdf =  boturdf
        self.floorurdf = floorurdf
        self.__goal_sampling = goal_sampling #sample a new goal after reset
        self.physicsClient = None
        #dim 0 to 4: Joint fault (multiplies to action sent)
        #dim 5 to 9: Joint angle measurement fault (fraction of 360 degree, adds to joint measurement)
        self.__mismatches = np.array(mismatches)
        self.__env_created = False
        self.__frameskip = int(controlStep/simStep)#frameskip
        self.__goal = np.array(goal)
        self.__goalBodyID = None
        
        self.render_mode = ''
        self._cam_dist = 3.3
        self._cam_yaw = 90.
        self._cam_pitch = -80.
        self._render_width = 1920
        self._render_height = int(self._render_width * 720/1280)
    
    def __compute_reward(self, goal):
        effPos = self.__endEffectorPos
        dist = np.linalg.norm(goal-effPos)
        return np.exp(-dist)
    
    def set_goal(self, goal=None):
        '''
        Sets random goal if not provided
        '''
        if goal is None:
            goal = np.random.rand(2)*2.-1. 
        self.__goal = np.zeros(2)
        self.__goal[0] = (goal[0] * 0.5 -1.0 ) * 0.6 # x: [0, -0.6]
        self.__goal[1] = goal[1] * 0.6 # y: [-0.6, 0.6]

    def reset(self):
        if self.__goal_sampling: self.set_goal() #set random goal if goal sampling is true
        if not self.__env_created:
            if self.render_mode=='human':
                self.physicsClient = self.p.connect(self.p.GUI)
            else:
                self.physicsClient = self.p.connect(self.p.DIRECT)
            self.p.setRealTimeSimulation(0, physicsClientId=self.physicsClient)
            self.p.resetSimulation(physicsClientId=self.physicsClient)
            self.p.setTimeStep(self.__simStep, physicsClientId=self.physicsClient)
            self.p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0,0,0], physicsClientId=self.physicsClient)
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)  
            self.p.setGravity(0,0,-10, physicsClientId=self.physicsClient) 
            self.visualGoalShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_CYLINDER, radius=0.08, length=0.01, 
                                                                visualFramePosition=[0.,0.,0.], 
                                                                visualFrameOrientation=self.p.getQuaternionFromEuler([0,0,0]) , 
                                                                rgbaColor=[0.0,0.0, 0.0, 0.5], 
                                                                specularColor=[0.5,0.5, 0.5, 1.0], 
                                                                physicsClientId=self.physicsClient)
            #Create the Arena
            bottom = np.array([1.6, 1.6+0.2, -1.6, 1.6])
            top = bottom + np.array([-2*1.6-0.2, -2*1.6-0.2, 0, 0])
            left = np.array([top[0], bottom[1], bottom[2]-0.2, bottom[2]])
            right = np.array([top[0], bottom[1], bottom[3], bottom[3]+0.2])
            Obstacles = np.array([bottom, top, left, right])
            add_obstacles(self, Obstacles)
            
            self.hexapodStartPos = [0,0,0.06] # Start at collision free state. Otherwise results becomes a bit random
            self.hexapodStartOrientation = self.p.getQuaternionFromEuler([0,0,0]) 
            flags= self.p.URDF_USE_INERTIA_FROM_FILE or self.p.URDF_USE_SELF_COLLISION or self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
            self.__botBodyId = self.p.loadURDF(self.boturdf, self.hexapodStartPos, self.hexapodStartOrientation, useFixedBase=1, flags=flags, physicsClientId=self.physicsClient) 
            self.__planeId = self.p.loadURDF(self.floorurdf, [0,0,0], self.p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.physicsClient)

            self.bodyId_activeJoints = []
            self.bodyId_links = []
            
            # JOINT_REVOLUTE=0 
            # JOINT_PRISMATIC=1 
            # JOINT_SPHERICAL=2 
            # JOINT_PLANAR=3 
            # JOINT_FIXED=4 
            
            for joint in range (self.p.getNumJoints(self.__botBodyId, physicsClientId=self.physicsClient)):
                info = self.p.getJointInfo(self.__botBodyId, joint, physicsClientId=self.physicsClient)
                if not info[2] == self.p.JOINT_FIXED:
                    self.bodyId_activeJoints.append({"bodyId": self.__botBodyId, 
                                                    "joint_name": info[1],
                                                    "joint_index": info[0],
                                                    "joint_type": info[2],
                                                    "link_name": info[12],
                                                    "link_index": info[-1],
                                                    "lower_limit": info[8],
                                                    "upper_limit": info[9],
                                                    "max_force": info[10],
                                                    "max_velocity": info[11]})
                if info[12] == b'fingertip': 
                    #NOTE Link index is equal to joint index. So "link_index": info[0]
                    self.bodyId_endEffector = {"bodyId": self.__botBodyId, "link_name": info[12], "link_index": info[0]}
            self.__env_created = True
            self.__jointPos=np.zeros(len(self.bodyId_activeJoints))
            self.__jointVel=np.zeros(len(self.bodyId_activeJoints))
            self.__endEffectorPos=np.zeros(2)
            self.__state = None
        
        if self.__goalBodyID is not None: 
            pos =  [self.__goal[0], self.__goal[1], 0.01]
            orient = self.p.getQuaternionFromEuler([0,0,0])
            self.p.resetBasePositionAndOrientation(self.__goalBodyID, pos, orient, physicsClientId=self.physicsClient)
        else:
            self.__goalBodyID = self.p.createMultiBody(baseMass=0.0,baseInertialFramePosition=[0,0,0], 
                                                        baseVisualShapeIndex = self.visualGoalShapeId, 
                                                        basePosition = [self.__goal[0], self.__goal[1], 0.01], 
                                                        useMaximalCoordinates=True, 
                                                        physicsClientId=self.physicsClient)
        
        for joint in self.bodyId_activeJoints:
            self.p.resetJointState(self.__botBodyId, joint["joint_index"], targetValue=0.0, targetVelocity=0.0, physicsClientId=self.physicsClient)   
        
        # Compute state:
        self.__update_state()
        return self.__state
    
    def step(self, action):
        '''
        state: 0 to 4: Joint position in degree
        state: 5 to 9: Joint velocities in radian/second
        state: 10 to 11: End effector (x,y) position
        '''
        # Set the command
        act = self.__mismatches[0:5] * action
        for i, joint in enumerate(self.bodyId_activeJoints):
            self.p.setJointMotorControl2(bodyUniqueId=joint["bodyId"], jointIndex=joint["joint_index"], 
                    controlMode=self.p.VELOCITY_CONTROL, 
                    targetVelocity = np.clip(act[i]*0.7, -0.7, 0.7),
                    force=200.0, 
                    physicsClientId=self.physicsClient)
        
        # Simulate for many steps
        for _ in range(self.__frameskip):
            self.p.stepSimulation(physicsClientId=self.physicsClient) 
            if self.p.getConnectionInfo(self.physicsClient)['connectionMethod'] == self.p.GUI:
                time.sleep(self.__simStep/float(self.__vspeed)) 

        # Compute state:
        self.__update_state()
        rew = self.__compute_reward(self.__goal)
        return self.state, rew, False, {}

    def render(self, mode='', close=False):
        self.render_mode = mode
        if not self.render_mode =='rgb_array' or not self.__env_created:
            return np.array([])       
        base_pos = [0., 0., 0.,]
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.p.getCameraImage(
        width = self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def __update_state(self):
        self.__compute_jointState()
        self.__compute_endEffectorState()
        self.__state = self.state

    def __compute_jointState(self):
        '''Must call at the end of Step and Reset'''
        for i, joint in enumerate(self.bodyId_activeJoints):
            self.__jointPos[i] = self.p.getJointState(joint["bodyId"],joint["joint_index"],physicsClientId=self.physicsClient)[0] 
            self.__jointVel[i] = self.p.getJointState(joint["bodyId"],joint["joint_index"],physicsClientId=self.physicsClient)[1] 
        
        self.__jointPos += self.__mismatches[-5::] * 2 * np.pi
        self.__jointPos = np.arctan2(np.sin(self.__jointPos), np.cos(self.__jointPos))
        self.__jointPos = np.rad2deg(self.__jointPos)
    
    def __compute_endEffectorState(self):
        '''Must call at the end of Step and Reset'''
        self.__endEffectorPos = np.array(self.p.getLinkState(self.bodyId_endEffector["bodyId"], self.bodyId_endEffector["link_index"], 
                                                    physicsClientId=self.physicsClient)[0])[0:2]

    @property
    def state(self):
        return np.concatenate((self.__jointPos, self.__jointVel, self.__endEffectorPos))
    
    def set_mismatch(self, mismatches):
        self.__mismatches = mismatches
    
    @property
    def goal(self):
        return self.__goal
    
if __name__ == '__main__':
    
    env = Reacher_env(visualizationSpeed=5.0, controlStep=0.3, goal_sampling=True)
    env.render("human")
    env.reset()
    pp = -np.zeros(5)
    mismatches = np.array([0,1.,1.,1.,1.,0.,0,0,0,0])
    env.set_mismatch(mismatches)
    for i in range(1000):
        pp = np.random.rand(5) * 2. - 1.
        s, r, _, _ = env.step(pp)
        base_ang = np.arctan2(np.sin(s[0]), np.cos(s[0]))
        print(r)
        if i%20==0:
            # env.set_goal()
            env.reset()
            print(env.goal)