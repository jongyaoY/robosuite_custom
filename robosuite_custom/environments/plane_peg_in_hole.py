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

from functools import partial

import numpy as np
from robosuite.environments.base import MujocoEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.buffers import RingBuffer
from robosuite.utils.observables import Observable, sensor

from robosuite_custom.models.objects import BlockSlot
from robosuite_custom.models.robots import Slider
from robosuite_custom.utils.corrupters import gaussian_corrupter
from robosuite_custom.utils.mujoco_utils import get_sensor_data


class PlanePegInHoleEnv(MujocoEnv):
    def __init__(
        self,
        **kwargs,
    ):
        self._ft_buffer_size = 2
        self._action_dim = 2
        self._input_max = 1.0
        self._rand_range = 0.1  # for slot offset
        self._force_noise = 0.5
        self._torque_noise = 0.1
        self._vel_noise = 0.005
        self._force_dim = 2
        self._torque_dim = 1
        self._tanh_scale = 1.0
        self.gym_keys = ["force", "torque", "velocity"]
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        empty_arena = EmptyArena()
        self.slot = BlockSlot()
        self.slider = Slider()
        self.slider.set_base_xpos([0, 0.3, 1])
        self.slot.set_pose([0, -0.15, 1])
        self.model = ManipulationTask(
            mujoco_arena=empty_arena,
            mujoco_robots=[self.slider],
            mujoco_objects=[self.slot],
        )

        self.model.save_model("2d_pih.xml")

    def _reset_internal(self):
        super()._reset_internal()  # robot.reset() is called here
        slider_id = self.sim.model.body_name2id(self.slider.root_body)
        self.sim.model.body_pos[slider_id] += np.concatenate(
            [
                self._rand_range * np.random.uniform(-1, 1, (1,)),
                self._rand_range / 2.0 * np.random.uniform(0, 1, (1,)),
                [0],
            ]
        )

    def _setup_references(self):
        super()._setup_references()
        self._force_indices = [0, 1]  # fx, fy
        self._force_sensor_name = (
            self.model.mujoco_robots[0].naming_prefix + "force_ee"
        )
        self._torque_sensor_name = (
            self.model.mujoco_robots[0].naming_prefix + "torque_ee"
        )
        self._accelerometer_name = (
            self.model.mujoco_robots[0].naming_prefix + "acc_ee"
        )
        self._torque_indices = [5]  # mz

    def _pre_action(self, action, policy_step=False):
        assert (
            len(action) == self.action_dim
        ), "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )
        action = np.clip(action, self.action_limits[0], self.action_limits[1])
        super()._pre_action(action)

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        # Action limits based on controller limits
        low = -self._input_max * np.ones(self.action_dim)
        high = self._input_max * np.ones(self.action_dim)

        return low, high

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = [], []

        lo, hi = self.action_limits
        low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    def _setup_observables(self):
        observables = super()._setup_observables()
        sensors = []
        names = []
        modality = "proprio"

        @sensor(modality=modality)
        def velocity(obs_cache):
            return np.array(
                self.sim.data.get_body_xvelp(self.slider.root_body)
            )[:2]

        @sensor(modality=modality)
        # tip pos in target frame
        def pos(obs_cache):
            # Not the bottom of the slot
            target_pos = self.sim.data.get_site_xpos(
                self.slot.target_site.get("name")
            )
            tip_pos = self.sim.data.get_site_xpos(self.slider.tip_site_name)
            return (tip_pos - target_pos)[:2]

        @sensor(modality=modality)
        def acceleration(obs_cache):
            acc = get_sensor_data(self._accelerometer_name, self.sim)
            return acc[:2]

        modality = "contact"

        @sensor(modality=modality)
        def force(obs_cache):
            force = get_sensor_data(self._force_sensor_name, self.sim)
            # force = self.sim.data.sensordata[self._force_indices]
            return force[:2]

        @sensor(modality=modality)
        def torque(obs_cache):
            torque = get_sensor_data(self._torque_sensor_name, self.sim)
            return torque[-1:]  # mz

        modality = "continuous"

        @sensor(modality=modality)
        def ft_continuous(obs_cache):
            if "ft_continuous_buffer" not in obs_cache:
                obs_cache["ft_continuous_buffer"] = RingBuffer(
                    dim=self._force_dim + self._torque_dim,
                    length=self._ft_buffer_size,
                )

            force = self.sim.data.sensordata[self._force_indices]

            torque = self.sim.data.sensordata[self._torque_indices]
            obs_cache["ft_continuous_buffer"].push(
                np.concatenate([force, torque])
            )
            return obs_cache["ft_continuous_buffer"].buf

        sensors += [
            force,
            torque,
            acceleration,
            ft_continuous,
            velocity,
            pos,
            velocity,
            force,
            torque,
        ]
        names += [
            "force",
            "torque",
            "acc_gt",
            "ft_continuous",
            "velocity",
            "pos_gt",
            "velocity_gt",
            "force_gt",
            "torque_gt",
        ]
        # self.gym_keys = names
        std_dict = {
            "force": self._force_noise * np.ones(self._force_dim),
            "torque": self._torque_noise * np.ones(self._torque_dim),
            "ft_continuous": np.concatenate(
                [
                    self._force_noise * np.ones(self._force_dim),
                    self._torque_noise * np.ones(self._torque_dim),
                ]
            ),
            "velocity": self._vel_noise * np.ones(2),
        }
        sample_freq = {
            "force": self.control_freq,
            "torque": self.control_freq,
            "acc_gt": self.control_freq,
            "force_gt": self.control_freq,
            "torque_gt": self.control_freq,
            "ft_continuous": self.control_freq * self._ft_buffer_size,
            "velocity": self.control_freq,
            "velocity_gt": self.control_freq,
            "pos_gt": self.control_freq,
        }
        for name, s in zip(names, sensors):
            try:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=sample_freq[name],
                    corrupter=(
                        partial(gaussian_corrupter, std=std_dict[name])
                        if name in std_dict
                        else None
                    ),
                )
            except KeyError as e:
                print(e)
        return observables

    def reward(self, action):
        # bottom_site_id = self.sim.model.site_name2id(self.slot.bottom_site.get('name'))
        bottom_pos = self.sim.data.get_site_xpos(
            self.slot.bottom_site.get("name")
        )
        tip_pos = self.sim.data.get_site_xpos(self.slider.tip_site_name)
        dist = np.linalg.norm(bottom_pos - tip_pos)

        reaching_reward = 1 - np.tanh(self._tanh_scale * dist)

        return reaching_reward
