from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import XmlBasedRobot
import pybullet
import numpy as np
import os,  inspect
import gym, gym.spaces, gym.utils, gym.utils.seeding
from pybullet_envs.bullet import bullet_client
from pkg_resources import parse_version


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

class BaseBulletEnv(gym.Env):
    """
    Base class for Bullet physics simulation environments in a Scene.
    These environments create single-player scenes and behave like normal Gym environments, if
    you don't use multiplayer.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    def __init__(self, robot, render=False):
        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.camera = Camera()
        self.isRender = render
        self.robot = robot
        self._seed()
        self._cam_dist = 10
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 200
        self._render_height = int(self._render_width * 720/1280)

        self.action_space = robot.action_space
        self.observation_space = robot.observation_space

    def configure(self, args):
        self.robot.args = args

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random # use the same np_randomizer for robot as for env
        return [seed]

    def _reset(self):
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        s = self.robot.reset(self._p)
        self.potential = self.robot.calc_potential()
        return s

    def _render(self, mode, close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self,'robot'):
            if hasattr(self.robot,'body_xyz'):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
        width = self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    def HUD(self, state, a, done):
        pass

    # backwards compatibility for gym >= v0.9.x
    # for extension of this class.
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    if parse_version(gym.__version__)>=parse_version('0.9.6'):
        close = _close
        render = _render
        reset = _reset
        seed = _seed


class Camera:
    def __init__(self):
        pass

    def move_and_look_at(self,i,j,k,x,y,z):
        lookat = [x,y,z]
        distance = 10
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)

class Scene:
    """A base class for single- and multiplayer scenes"""

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.np_random, seed = gym.utils.seeding.np_random(None)
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

        self.test_window_still_open = True  # or never opened
        self.human_render_detected = False  # if user wants render("human"), we open test window

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer: return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def episode_restart(self, bullet_client):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.clean_everything()
        #self.cpp_world.test_window_history_reset()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step(self.frame_skip)


class SingleRobotEmptyScene(Scene):
    multiplayer = False  # this class is used "as is" for InvertedPendulum, Reacher


class World:

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.clean_everything()

    def clean_everything(self):
        # p.resetSimulation()
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setDefaultContactERP(0.9)
        # print("self.numSolverIterations=",self.numSolverIterations)
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=self.numSolverIterations, numSubSteps=self.frame_skip)

    def step(self, frame_skip):
        self._p.stepSimulation()

class StadiumScene(Scene):
    multiplayer = False
    zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen   = 105*0.25	# FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25	 # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if self.stadiumLoaded == 0:
            self.stadiumLoaded = 1
            # print(os.path.join(os.path.dirname(__file__), "assets", "scenes", "stadium", "plane_stadium.sdf"))
            filename = os.path.join(os.path.dirname(__file__),"assets", "scenes", "stadium", "plane_stadium.sdf")
            self.ground_plane_mjcf=self._p.loadSDF(filename)
            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i,-1,rgbaColor=[1,1,1,0.8])
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,1)

class WalkerBaseMuJoCoEnv(BaseBulletEnv):
    def __init__(self, robot, render=False):
        # print("WalkerBase::__init__")
        BaseBulletEnv.__init__(self, robot, render)
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId=-1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return self.stadium_scene

    def reset(self):
        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = BaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId=self._p.saveState()
        #print("saving state self.stateId:",self.stateId)
        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost	 = -2.0	 # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost	= -0.1	 # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0	 # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1	 # discourage stuck joints

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        #electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        #electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            #print("electricity_cost")
            #print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            alive,
            progress,
            #electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

class MJCFBasedRobot(XmlBasedRobot):
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True, add_ignored_joints=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision, add_ignored_joints)
        self.model_xml = model_xml
        self.doneLoading=0

    def reset(self, bullet_client):

        # full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mjcf", self.model_xml)
        full_path = os.path.join(os.path.dirname(__file__), "assets", "mjcf", self.model_xml)

        self._p = bullet_client
        #print("Created bullet_client with id=", self._p._client)
        if self.doneLoading == 0:
            self.ordered_joints = []
            self.doneLoading=1
            if self.self_collision:
                self.objects = self._p.loadMJCF(full_path, flags=pybullet.URDF_USE_SELF_COLLISION|pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(full_path)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        return s

    def calc_potential(self):
        return 0

class Ant(WalkerBase, MJCFBasedRobot):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        MJCFBasedRobot.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=27)

    def calc_state(self):
        WalkerBase.calc_state(self)
        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()  # shape (15,)

        velocity = self.parts['torso'].get_velocity()
        qvel = np.hstack((velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()  # shape (14,)
        
        return np.concatenate([
            qpos.flat[2:],                   # self.sim.data.qpos.flat[2:],
            qvel.flat						 # self.sim.data.qvel.flat,
        ])

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
    
    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for key in self.jdict.keys():
            self.jdict[key].power_coef = 400.0 #joint effort

class AntMuJoCoEnv(WalkerBaseMuJoCoEnv):
    def __init__(self, mismatches=[1.,1.,1.,1.,1.,1.,1.,1.,0.]):
        self.robot = Ant()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot) 
        self.__mismatch = np.array(mismatches)
        # self.__mismatch = np.ones(9)
        # self.__mismatch[8] = 0.0
        self.__ant_state = None
    
    def step(self, a):
        act = self.__mismatch[0:8] * a
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(act)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        #electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        #electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            #print("electricity_cost")
            #print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        # self.rewards = [
        #     alive,
        #     progress,
        #     #electricity_cost,
        #     joints_at_limit_cost,
        #     feet_collision_cost
        # ]
        self.rewards = [
            # alive,
            # state[0]*0.01, #height
            state[13] # x lelocity
            #electricity_cost,
            # joints_at_limit_cost,
            # feet_collision_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        bodyQnionPose = state[1:5]
        bodyEulerPose = np.array(pybullet.getEulerFromQuaternion(bodyQnionPose))
        bodyEulerPose[2] = 2 * np.pi * self.__mismatch[8] #perturbing z rotation
        state[1:5] = np.array(pybullet.getQuaternionFromEuler(bodyEulerPose))
        self.__ant_state=state
        return state, sum(self.rewards), bool(done), {}
    
    def __str__(self):
        s = "\n---------Ant environment object---------------\n"
        s += "Custom Ant Environment similar to Mujoco's env, but with pybullet.\n"
        s += "Action dim 8 (for 8 joints)\n"
        s += "Observation dim 27 (Reduced from 111 dim since remaining values are zero)\n"
        s += "\ndim 0           : Height\n"
        s += "dim 1 to dim 4  : Body pose (Quaternion)\n"
        s += "dim 5 to dim 12 : Joint positions\n"
        s += "dim 13 to dim 15: body x,y,z linear velocity (verify!!)\n"
        s += "dim 16 to dim 18: body x,y,z angular velocity (verify!!)\n"
        s += "dim 19 to dim 26: joint velocities\n"
        s += "---------------------------------------------\n"
        return s


    def get_reward(self, prev_obs, action, next_obs):
        return next_obs[13]
    
    def set_mismatch(self, mismatch):
        '''
        Set joint torque multiplier in [-1, 1] and 
        set orientation perturbation in [-1, 1] as fraction 
        of full rotation along z axis (vertical axis).
        dim 0 to 7 : Joint torque multipliers  
        dim 8      : Orientation perturbation  
        '''
        assert len(mismatch) == 9, "Mismatch must be 9 dimensional. Check doc string."
        self.__mismatch = mismatch
    
    @property
    def state(self):
        return self.__ant_state
    
    def reset(self):
        state = super(AntMuJoCoEnv, self).reset()
        bodyQnionPose = state[1:5]
        bodyEulerPose = np.array(pybullet.getEulerFromQuaternion(bodyQnionPose))
        bodyEulerPose[2] = 2 * np.pi * self.__mismatch[8] #perturbing z rotation
        state[1:5] = np.array(pybullet.getQuaternionFromEuler(bodyEulerPose))
        self.__ant_state = state
        return self.state

if __name__ == "__main__":
    import gym
    import time
    import fast_adaptation_embedding.env
    system = gym.make("AntMuJoCoEnv_fastAdapt-v0")
    system.set_mismatch(np.array([1.,1,1,1,1,1,1,1, -0.8]))
    system.render(mode="human")
    import time
    s = system.reset()
    rew = 0
    for i in range(200):
        _, r , _, _ = system.step(system.action_space.sample())
        rew +=r
        time.sleep(0.01)
    print(rew)
# /home/rkaushik/projects/fast_adaptation_embedding/arm_data/best_action_seq_ant.npy