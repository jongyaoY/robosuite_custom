# MIT License
#
# Copyright (c) [2024] [Zongyao Yi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# flake8: noqa


import math
from functools import partial

import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TestBedArena
from robosuite.models.objects import EVChagerSocket, SocketBoard
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.buffers import RingBuffer
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.observables import Observable, sensor
from scipy.spatial.transform import Rotation as R

import robosuite_custom
from robosuite_custom.utils.corrupters import gaussian_corrupter
from robosuite_custom.utils.inverse_kinematics import qpos_from_site_pose
from robosuite_custom.utils.placement_samplers import (
    UniformRandomSamplerInLocalFrame,
)
from robosuite_custom.utils.point_clouds import (
    generate_point_cloud,
    get_motion_field,
    project_to_cam,
)

# Default peg in hole environment configuration
DEFAULT_PIH_CONFIG = {
    # env settings
    "randomize_arena": True,
    # settings for reward
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "ee_accel_penalty": 0.1,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold
    "excess_torque_penalty_mul": 1.0,  # penalty for each step that the torque is over the safety threshold
    "socket_contact_reward": 0.01,
    "task_complete_reward": 9.0,
    "time_penalty": 1.0,
    "reward_weights": {
        "reaching": 3,
        "orientation": 0,
        "contact": 0,
    },
    "collision_check": False,
    "tanh_scale": 1.0,
    # settings for thresholds
    "contact_threshold": 1.0,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 0.5,  # force threshold (N) to overcome to get increased contact reward
    "pressure_threshold_max": 10.0,  # maximum force allowed (N)
    "torque_threshold_max": 0.05,  # maximum torque allowed (Nm)
    # misc settings
    "get_info": True,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": False,  # Whether we allow for early terminations or not
    "ft_buffer_size": 2,
    # for socket observation
    "noisy_socket_obs": True,
    "socket_obs_bias_range": {
        "x_range": 0.001 * np.array([-1, 1]),
        "y_range": 0.001 * np.array([-1, 1]),
        "z_range": 0.001 * np.array([-1, 1]),
        "ax_range": 0.0 * np.array([-1, 1]),
        "ay_range": 0.0 * np.array([-1, 1]),
        "az_range": 0.0 * np.array([-1, 1]),
    },
    "socket_obs_var": {
        "x": 0.001,
        "y": 0.001,
        "z": 0.001,
        "ax": 0.00,
        "ay": 0.00,
        "az": 0.00,
    },
    # controller settings
    "action_scale": 1.0,
    # randomization settings
    "randomization_opts": {
        "socket": {
            # "rotation_axis": None,
            "rotation_axis": "xy",
            "rotation": np.array([-15, 15])
            * np.pi
            / 180.0,  # box size in x, y, z axis
            "x_range": np.array([-0.1, 0.1]),
            "y_range": np.array([-0.1, 0.1]),
        },
        "robot": {
            "predock_offset": np.array([0, 0, 0.06]),
            "rotation": np.array([0, 0]) * np.pi / 180.0,
            # "rotation": np.array([-15, 15])* np.pi / 180.,
            "x_range": np.array([-0.01, 0.01]),
            "y_range": np.array([-0.01, 0.01]),
            "z_range": np.zeros(2),
        },
    },
}


class PegInHoleEnv(SingleArmEnv):
    """
    This class corresponds to the peg-in-hole task for a single robot arm

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default ("WipingGripper") will raise an
            AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default configuration dict at the top of this file.
            If None is specified, the default configuration will be used.

        Raises:
            AssertionError: [Gripper specified]
            AssertionError: [Bad reward specification]
            AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="EVChargerPlug",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="sideview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        task_config=None,
        renderer="mujoco",
        renderer_config=None,
        save_model=False,
    ):
        # Assert that the gripper type is None
        assert gripper_types in [
            "PlugGripper",
            "EVChargerPlug",
        ], "Trying to specify gripper other than PlugGripper in PegInHole environment!"
        assert isinstance(robots, str), "Only single robot is supported"
        assert robots in ["UR10e", "UR5e"], "Now only UR robots are suppoerted"
        # Settings for debug
        self.save_model = save_model

        # Get config
        self.task_config = (
            task_config if task_config is not None else DEFAULT_PIH_CONFIG
        )

        # Set task-specific parameters
        self.randomize_arena = self.task_config["randomize_arena"]
        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config[
            "arm_limit_collision_penalty"
        ]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config[
            "excess_force_penalty_mul"
        ]
        self.excess_torque_penalty_mul = self.task_config[
            "excess_torque_penalty_mul"
        ]
        self.socket_contact_reward = self.task_config["socket_contact_reward"]
        self.reward_weights = self.task_config["reward_weights"]
        self.tanh_scale = self.task_config["tanh_scale"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        self.task_complete_reward = self.task_config[
            "task_complete_reward"
        ]  # TODO
        self.time_penalty = (
            self.task_config["time_penalty"]
            if self.task_config["early_terminations"]
            else 0
        )
        self.reward_normalization_factor = 1.0 / (
            self.reward_weights["reaching"] * 1.0
            + self.reward_weights["orientation"] * 3
            + self.reward_weights["contact"] * self.socket_contact_reward
            + self.task_complete_reward
            - self.time_penalty
        )  # TODO
        # options for domain rand
        self.rand_opts = self.task_config["randomization_opts"]
        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config[
            "pressure_threshold_max"
        ]
        self.torque_threshold_max = self.task_config["torque_threshold_max"]
        # misc settings
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.collision_check = self.task_config["collision_check"]
        # socket observation
        self.noisy_socket_obs = self.task_config["noisy_socket_obs"]
        self.socket_obs_noise = {
            "bias": np.zeros(6),
            "var": 0.02 * np.ones(6),
        }
        self.socket_obs_bias_range = self.task_config["socket_obs_bias_range"]
        self.socket_obs_var = self.task_config["socket_obs_var"]
        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        # Socket randomization
        self.socket = None
        self.socket_board = None
        self.socket_refence_pos = None
        self.socket_refence_rot = None
        self.socket_board_offsets = np.array([0, 0, -0.018])
        self.placement_initializer = UniformRandomSamplerInLocalFrame(
            name="SocketSampler",
            x_range=self.rand_opts["socket"]["x_range"],
            y_range=self.rand_opts["socket"]["y_range"],
            rotation_axis=self.rand_opts["socket"]["rotation_axis"],
            rotation=self.rand_opts["socket"]["rotation"],
        )
        self.predock_offest = self.rand_opts["robot"][
            "predock_offset"
        ]  # TODO Predock position of the robot, relative to the socket
        # set other insertion-specific attributes
        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0
        self.metadata = []
        self.spec = "spec"

        # action scale
        self.action_scale = self.task_config["action_scale"]
        # *************************#
        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs
        # observable keys for gym env
        self.gym_keys = []
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.socket_site_name = self.model.mujoco_objects[0].important_sites[
            "target"
        ]
        self.socket_site_id = self.sim.model.site_name2id(
            self.socket_site_name
        )
        self.plug_site_name = self.robots[0].gripper.important_sites[
            "grip_site"
        ]
        self.plug_site_id = self.robots[0].eef_site_id

        # self.geom_id_to_bodies = generate_geom_to_body_mapping(self.sim)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix
        sensors = []
        names = []
        modality = f"{pf}contact"

        @sensor(modality=modality)
        def force(obs_cache):
            return self.robots[0].ee_force - self.ee_force_bias

        @sensor(modality=modality)
        def torque(obs_cache):
            return self.robots[0].ee_torque - self.ee_torque_bias

        modality = f"{pf}continuous"

        @sensor(modality=modality)
        def ee_forces_continuous(obs_cache):
            if "ee_forces_continuous" not in obs_cache:
                obs_cache["ee_forces_continuous"] = RingBuffer(
                    dim=6, length=self.task_config["ft_buffer_size"]
                )
            else:
                ft_data = np.concatenate(
                    [
                        self.robots[0].ee_force - self.ee_force_bias,
                        self.robots[0].ee_torque - self.ee_torque_bias,
                    ]
                )
                obs_cache["ee_forces_continuous"].push(ft_data)
            return obs_cache["ee_forces_continuous"].buf

        modality = f"{pf}binary"

        @sensor(modality=modality)
        def contact(obs_cache):
            if "contact" not in obs_cache:
                obs_cache["contact"] = RingBuffer(
                    dim=1, length=self.task_config["ft_buffer_size"]
                )
            else:
                obs_cache["contact"].push(self._has_gripper_contact)
            return np.squeeze(obs_cache["contact"].buf, axis=1)

        modality = "proprio"

        @sensor(modality=modality)
        def eef_vel(obs_cache):
            return self.robots[0].recent_ee_vel.current

        @sensor(modality=modality)
        def delta_pos(obs_cache):
            socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
            plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
            return socket_pos - plug_pos

        @sensor(modality=modality)
        def eef_pos(obs_cache):
            plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
            return plug_pos

        @sensor(modality=modality)
        def eef_rot(obs_cache):
            plug_mat = self.sim.data.get_site_xmat(self.plug_site_name)
            return R.from_matrix(plug_mat).as_rotvec()

        @sensor(modality=modality)
        def socket_pose_in_ee(obs_cache):
            socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
            socket_mat = self.sim.data.get_site_xmat(self.socket_site_name)
            plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
            plug_mat = self.sim.data.get_site_xmat(self.plug_site_name)
            wTp = T.make_pose(plug_pos, plug_mat)
            wTs = T.make_pose(socket_pos, socket_mat)
            pTs = np.matmul(T.pose_inv(wTp), wTs)
            socket_pos_in_ee = pTs[:3, 3]
            socket_rot_in_ee = R.from_matrix(pTs[:3, :3]).as_rotvec()
            return np.concatenate([socket_pos_in_ee, socket_rot_in_ee])

        @sensor(modality=modality)
        def socket_pose(obs_cache):
            socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
            socket_mat = self.sim.data.get_site_xmat(self.socket_site_name)
            socket_rot = R.from_matrix(socket_mat).as_rotvec()
            return np.concatenate([socket_pos, socket_rot])

        sensors += [
            force,
            torque,
            ee_forces_continuous,
            eef_vel,
            delta_pos,
            eef_pos,
            eef_rot,
            contact,
            socket_pose,
            socket_pose_in_ee,
        ]
        names += [
            f"{pf}force",
            f"{pf}torque",
            f"{pf}ee_forces_continuous",
            f"{pf}eef_vel",
            f"{pf}delta_pos",
            f"{pf}eef_pos",
            f"{pf}eef_rot",
            f"{pf}contact",
            "socket_pose",
            "socket_pose_in_ee",
        ]
        # self.gym_keys = names
        self.gym_keys = [
            f"{pf}force",
            f"{pf}torque",
            f"{pf}delta_pos",
            f"{pf}eef_vel",
        ]  # Select which observables for RL
        # self.gym_keys = [f"{pf}force", f"{pf}torque", f"{pf}eef_vel", f"{pf}eef_pos", f"{pf}eef_rot", "socket_pose_in_ee"] # Select which observables for RL
        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )
            if name == f"{pf}ee_forces_continuous" or name == f"{pf}contact":
                observables[name].set_sampling_rate(
                    int(self.control_freq * self.task_config["ft_buffer_size"])
                )
            if name == f"{pf}force":
                std = np.array(
                    [
                        0.055,  # fx
                        0.055,  # fy
                        0.040,  # fz
                    ]
                )
                observables[name].set_corrupter(
                    partial(gaussian_corrupter, std=std)
                )
            if name == f"{pf}torque":
                std = np.array([0.0012, 0.0012, 0.0005])  # mx  # my  # mz
                observables[name].set_corrupter(
                    partial(gaussian_corrupter, std=std)
                )
            if name == f"{pf}ee_forces_continuous":
                std = np.array(
                    [
                        0.055,  # fx
                        0.055,  # fy
                        0.040,  # fz
                        0.0012,  # mx
                        0.0012,  # my
                        0.0005,  # mz
                    ]
                )
                observables[name].set_corrupter(
                    partial(gaussian_corrupter, std=std)
                )

        if self.noisy_socket_obs:
            self._set_socket_pose_corrupter(observables)
            # corrupter = partial(
            #     gaussian_corrupter,
            #     bias=self.socket_obs_noise["bias"],
            #     std=self.socket_obs_noise["var"],
            #     )
            # observables["socket_pose_in_ee"].set_corrupter(corrupter)
            # observables["socket_pose"].set_corrupter(corrupter)

        return observables

    def _create_camera_sensors(
        self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"
    ):
        sensors, names = super()._create_camera_sensors(
            cam_name, cam_w, cam_h, cam_d, cam_segs, modality
        )
        sensor_name = f"{cam_name}_optflow"
        # depth_sensor_name = f"{cam_name}_depth"

        @sensor(modality=modality)
        def optical_flow(obs_cache):
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

            seg, depth = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=True,
                segmentation=True,
            )
            seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
            depth = depth[::convention]
            point_cloud = generate_point_cloud(
                depth, self.sim, cam_name, cam_w, cam_h
            )
            point_cloud_labels = np.fromiter(
                map(lambda x: self.sim.model.geom_bodyid[x], seg.flatten()),
                dtype=np.int32,
            ).reshape(cam_h, cam_w, 1)
            motion_field = get_motion_field(
                point_cloud,
                point_cloud_labels,
                self.sim,
                cam_name,
            )
            cam_id = self.sim.model.camera_name2id(cam_name)
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = cam_h / (2 * math.tan(fovy / 2.0))

            optflow = project_to_cam(point_cloud, motion_field, f)

            # fake_optiflow = motion_field[...,:2]
            return optflow

        sensors.append(optical_flow)
        names.append(sensor_name)

        return sensors, names

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms
        mujoco_arena = TestBedArena()
        socket = EVChagerSocket("Socket")
        socket_board = SocketBoard(
            "Board", inner_size=[0.08, 0.11], outter_size=[0.5, 0.6]
        )
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.socket_refence_pos = mujoco_arena.table_top_abs + np.array(
            [0.3, 0, 0.5]
        )  # TODO move to config
        self.socket_refence_rot = R.from_euler(
            "XYZ", [90, -90, 0], degrees=True
        ).as_rotvec()

        socket.set_pose(self.socket_refence_pos, self.socket_refence_rot)
        socket_board.set_pose(
            self.socket_refence_pos + self.socket_board_offsets,
            self.socket_refence_rot,
        )
        self.socket = socket
        self.socket_board = socket_board
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()
        self.placement_initializer.add_objects(socket)
        self.placement_initializer.reference_pos = socket.init_pos

        # Adjust base pose accordingly
        xpos = mujoco_arena.table_mount_pos
        robot_base_offset = np.array([0.1, 0.1, 0.0])
        self.robots[0].robot_model.set_base_xpos(xpos + robot_base_offset)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[socket, socket_board],
        )

        if self.save_model:
            self.model.save_model("peg_in_hole_model.xml")

    def reward(self, action):
        """
        Reward function for the task.
        Sparse un-normalized reward:
            - Task complete reward: task_complete_reward
        Un-normalized summed components if using reward shaping:
            see _stage_reward

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        reward = self._simple_reward()
        normalize_factor = 1.0
        if self._check_success():
            reward += self.task_complete_reward
        return self.reward_scale * reward * normalize_factor

    def _simple_reward(self):
        reward = 0
        socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
        plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
        dist = np.linalg.norm(socket_pos - plug_pos)
        reaching_reward = 1 - np.tanh(self.tanh_scale * dist)
        reward += reaching_reward
        return reward

    def _stage_reward(self):
        """
        Stage reward function
            - Penalty for collision { arm_limit_collision_penalty, 0 }
            - #TODO Penalty for contact between peg and table
            - Penalty exceeding joint limits { arm_limit_collision_penalty, 0 }
            - Reaching reward: in [0, 1]
            - Orientation reward: in [0, 3]
            - Contact reward: in {0, contact_reward_weight*socket_contact_reward}
            - Torque penalty: in [-inf, 0]
            - Excessive force penalty: in [-inf, 0] scaled by eef force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0] scaled by estimated eef acceleration and directly
              proportional to self.ee_accel_penalty
            - Time penalty: 4
        Max reward: reaching_weight*1 + orientation_weight*3 + contact_reward_weight*socket_contact_reward + task_complete_reward - time_penalty

        Returns:
            float: reward value

        """
        reward = 0

        total_force_ee = np.linalg.norm(
            np.array(self.robots[0].recent_ee_forcetorques.current[:3])
        )
        total_torque_ee = np.linalg.norm(
            np.array(self.robots[0].recent_ee_forcetorques.current[3:])
        )
        # Get socket pose
        socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
        socket_mat = self.sim.data.get_site_xmat(self.socket_site_name)
        plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
        plug_mat = self.sim.data.get_site_xmat(self.plug_site_name)
        # Get variables
        t, d, cos = self._compute_orientation(
            plug_pos, plug_mat, socket_pos, socket_mat
        )

        # Reaching reward
        dist = np.linalg.norm(socket_pos - plug_pos)
        reaching_reward = 1 - np.tanh(self.tanh_scale * dist)
        reward += self.reward_weights["reaching"] * reaching_reward

        # Orientation reward
        orientation_reward = 0
        orientation_reward += 1 - np.tanh(self.tanh_scale * d)
        orientation_reward += 1 - np.tanh(self.tanh_scale * np.abs(t))
        orientation_reward += cos

        reward += self.reward_weights["orientation"] * orientation_reward
        # Reward for keeping peg in contact with hole
        if self.sim.data.ncon != 0 and self._has_gripper_contact:
            reward += (
                self.reward_weights["contact"] * self.socket_contact_reward
            )

        # Torque penalty
        if total_torque_ee > self.torque_threshold_max:
            reward -= self.excess_torque_penalty_mul * total_torque_ee
            self.t_excess += 1

        # Excessive force penalty
        if total_force_ee > self.pressure_threshold_max:
            reward -= self.excess_force_penalty_mul * total_force_ee
            self.f_excess += 1
        # TODO Reward for pressing the socket?
        elif (
            total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1
        ):
            pass

        # Large acceleration penalty
        reward -= self.ee_accel_penalty * np.mean(
            abs(self.robots[0].recent_ee_acc.current)
        )

        # Time penalty
        reward -= self.time_penalty

        return reward

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
        plug_pos = self.sim.data.get_site_xpos(self.plug_site_name)
        socket_mat = self.sim.data.get_site_xmat(self.socket_site_name)
        plug_mat = self.sim.data.get_site_xmat(self.plug_site_name)
        t, _, _ = self._compute_orientation(
            plug_pos, plug_mat, socket_pos, socket_mat
        )
        return np.abs(t) < 0.0001

    def _check_terminated(self, info=None):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (insertion succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            if info is not None:
                info.update({"termination_info": "collision"})
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if info is not None:
                info.update({"termination_info": "success"})
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.robots[0].check_q_limits():
            if info is not None:
                info.update({"termination_info": "joint limits reached"})
            terminated = True

        return terminated

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)
        if self.get_info:
            socket_pos = self.sim.data.site_xpos[self.socket_site_id]
            socket_mat = self.sim.data.site_xmat[self.socket_site_id]
            plug_pos = self.sim.data.site_xpos[self.plug_site_id]
            plug_mat = self.sim.data.site_xmat[self.plug_site_id]

            t, d, cos = self._compute_orientation(
                plug_pos, plug_mat, socket_pos, socket_mat
            )

            info["target_residual"] = np.linalg.norm(socket_pos - plug_pos)
            info["parallel_distance"] = t
            info["perpendicular_distance"] = d
            info["angle"] = cos
            info["collision"] = self.collisions
            info["success"] = self._check_success()
            info["terminated"] = self._check_terminated()

        return reward, done, info

    def _compute_orientation(self, peg_pos, peg_mat, hole_pos, hole_mat):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle


                v
                ^   d      center - peg_pos
                ----------
                |       *
                |      *
            t   |     *
                |    *
                |   *
                | *
        """
        peg_mat.shape = (3, 3)
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center_in_hole_frame = np.array([0, 0, 0])
        center = hole_pos + hole_mat @ center_in_hole_frame

        t = np.dot(center - peg_pos, v)
        d = np.linalg.norm(np.cross(v, peg_pos - center))

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(np.dot(hole_normal, v)),
        )

    def _reset_internal(self):
        super()._reset_internal()  # robot.reset() is called here
        # inherited class should reset positions of objects (only if we're not using a deterministic reset)
        if self.randomize_arena:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                obj_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[obj_id] = obj_pos
                self.sim.model.body_quat[obj_id] = T.convert_quat(
                    obj_quat, to="wxyz"
                )
            # Set socket board position according to the socket
            socket_id = self.sim.model.body_name2id(self.socket.root_body)
            socket_board_id = self.sim.model.body_name2id(
                self.socket_board.root_body
            )
            self.sim.model.body_pos[socket_board_id] = (
                self.sim.model.body_pos[socket_id] + self.socket_board_offsets
            )
            self.sim.model.body_quat[socket_board_id] = (
                self.sim.model.body_quat[socket_id]
            )
            self.sim.forward()
        socket_pos = self.sim.data.get_site_xpos(self.socket_site_name)
        socket_mat = self.sim.data.get_site_xmat(self.socket_site_name)
        ori_socket = R.from_matrix(socket_mat)
        ori_offset = R.from_euler("xzy", [np.pi, -np.pi / 2.0, 0])
        goal_pose = np.zeros(6)
        # goal_noise = 0.01*np.random.uniform(-np.ones(3), np.ones(3)) #TODO
        # goal_noise[2] = 0
        pos_noise = np.array(
            [
                np.random.uniform(*self.rand_opts["robot"]["x_range"]),
                np.random.uniform(*self.rand_opts["robot"]["y_range"]),
                np.random.uniform(*self.rand_opts["robot"]["z_range"]),
            ]
        )
        rot_noise_angle = np.random.uniform(
            *self.rand_opts["robot"].get("rotation", np.zeros(2))
        )
        rot_noise_axis = np.random.uniform(-np.ones(3), np.ones(3))
        if np.linalg.norm(rot_noise_axis) != 0:
            rot_noise_axis /= np.linalg.norm(rot_noise_axis)
        rot_noise = R.from_rotvec(rot_noise_angle * rot_noise_axis)
        goal_pose[:3] = socket_pos + socket_mat @ (
            self.predock_offest + pos_noise
        )
        goal_pose[3:] = (rot_noise * ori_socket * ori_offset).as_rotvec()
        qpos = qpos_from_site_pose(
            sim=self.sim,
            site_name=self.plug_site_name,
            goal_pose=goal_pose,
            inplace=False,
        )
        self.robots[0].init_qpos = qpos
        self.robots[0].reset(deterministic=True)

        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0

        # sample socket bias
        if self.noisy_socket_obs:
            self.socket_obs_noise["bias"] = np.array(
                [
                    np.random.uniform(*self.socket_obs_bias_range["x_range"]),
                    np.random.uniform(*self.socket_obs_bias_range["y_range"]),
                    np.random.uniform(*self.socket_obs_bias_range["z_range"]),
                    np.random.uniform(*self.socket_obs_bias_range["ax_range"]),
                    np.random.uniform(*self.socket_obs_bias_range["ay_range"]),
                    np.random.uniform(*self.socket_obs_bias_range["az_range"]),
                ]
            )
            self.socket_obs_noise["var"] = np.array(
                [
                    self.socket_obs_var["x"],
                    self.socket_obs_var["y"],
                    self.socket_obs_var["z"],
                    self.socket_obs_var["ax"],
                    self.socket_obs_var["ay"],
                    self.socket_obs_var["az"],
                ]
            )
            self._set_socket_pose_corrupter()

    def _set_socket_pose_corrupter(self, observables=None):
        if observables is not None:
            corrupter = partial(
                gaussian_corrupter,
                bias=self.socket_obs_noise["bias"],
                std=self.socket_obs_noise["var"],
            )
            observables["socket_pose_in_ee"].set_corrupter(corrupter)
            observables["socket_pose"].set_corrupter(corrupter)

        elif "socket_pose" in self._observables:
            corrupter = partial(
                gaussian_corrupter,
                bias=self.socket_obs_noise["bias"],
                std=self.socket_obs_noise["var"],
            )
            self.modify_observable(
                "socket_pose",
                "corrupter",
                corrupter,
            )
            self.modify_observable(
                "socket_pose_in_ee",
                "corrupter",
                corrupter,
            )

    def _hole_pose_in_peg_frame(self, peg_pos, peg_rot, hole_pos, hole_rot):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pose_in_world = T.make_pose(peg_pos, peg_rot)

        # World frame
        hole_pose_in_world = T.make_pose(hole_pos, hole_rot)

        # world_pose_in_hole = T.pose_inv(hole_pose_in_world)
        world_pose_in_peg = T.pose_inv(peg_pose_in_world)

        # peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)
        hole_pose_in_peg = T.pose_in_A_to_pose_in_B(
            hole_pose_in_world, world_pose_in_peg
        )
        return hole_pose_in_peg

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        return (
            np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
            > self.contact_threshold
        )
