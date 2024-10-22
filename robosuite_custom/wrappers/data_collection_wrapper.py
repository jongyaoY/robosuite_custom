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

"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import os
from datetime import datetime

import cv2 as cv
import h5py
import numpy as np
from robosuite.wrappers import Wrapper
from scipy.spatial.transform import Rotation as R


def visualize_flow(flow, shape):
    hsv = np.zeros(shape=shape, dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


class DataCollectionWrapper(Wrapper):
    def __init__(self, env, directory, flush_freq=50, env_id=0, chunk=1000):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory
        self.env_id = env_id
        self.total_timesteps = 0
        self.chunk = chunk
        self.dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        # in-memory cache for simulation states and action info
        self.keys = [
            "action",
            "ee_forces_continuous",
            "image",
            "depth_data",
            "optical_flow",
            "contact",
            "proprio",
        ]
        self.dataset = {}
        for key in self.keys:
            self.dataset.update({key: []})
        self.init_observations = None
        self.action_infos = []  # stores information about actions taken
        self.successful = False  # stores success state of demonstration

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print(
                "DataCollectionWrapper: making new directory at {}".format(
                    directory
                )
            )
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__

        group_id = (self.total_timesteps // self.chunk) % 10
        flush_id = (self.total_timesteps % self.chunk) // self.flush_freq
        h5_file = os.path.join(
            self.directory,
            "{}_{}_{}_{}_{}_{}.h5".format(
                env_name,
                self.env_id,
                self.dt_string,
                group_id,
                flush_id,
                self.chunk,
            ),
        )

        dataset = h5py.File(h5_file, "w")
        for key in self.dataset.keys():
            if len(self.dataset[key]) == 0:
                continue
            dtype = self.dataset[key][0].dtype
            data = np.array(self.dataset[key], dtype=dtype)
            if len(data.shape) == 1:
                chunks = (1,)
            elif len(data.shape) == 2:
                chunks = (1, data.shape[1])
            else:
                chunks = tuple([1]) + data.shape[1:]
            # chunks = (1, data.shape[1]) if len(data.shape) < 3 else tuple([1]) + data.shape[1:]
            dset = dataset.create_dataset(
                key, data.shape, chunks=chunks, dtype=dtype
            )
            dset[...] = data
            self.dataset[key].clear()

        dataset.close()
        # self.action_infos = []
        # self.successful = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self.init_observations = ret
        self._start_new_episode()

        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        self.t += 1
        self.total_timesteps += 1
        if self.total_timesteps % (self.chunk * 10) == 0:
            self.dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # Copy neccessary data
        observations = ret[0]
        camera_name = self.camera_names[0] + "_"
        pf = self.robots[0].robot_model.naming_prefix
        self.dataset["action"].append(np.array(action))
        for key in self.keys:
            if pf + key in observations:
                self.dataset[key].append(observations[pf + key])
            elif key == "image":
                self.dataset["image"].append(observations[camera_name + key])
            elif key == "depth_data":
                self.dataset[key].append(observations[camera_name + "depth"])
            elif key == "optical_flow":
                self.dataset["optical_flow"].append(
                    observations[camera_name + "optflow"]
                )
            elif key == "proprio":
                ee_pos = observations[pf + "eef_pos"]
                ee_ori = R.from_quat(observations[pf + "eef_quat"]).as_rotvec()
                ee_vel = observations[pf + "eef_vel"][:3]
                ee_vel_ori = observations[pf + "eef_vel"][3:]
                proprio = np.concatenate([ee_pos, ee_ori, ee_vel, ee_vel_ori])
                self.dataset[key].append(proprio)

        # flush collected data to disk if necessary
        if self.t % (self.flush_freq - 1) == 0:
            self._flush()

        return ret
