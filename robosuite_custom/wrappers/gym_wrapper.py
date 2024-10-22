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

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robosuite.wrappers import Wrapper

from robosuite_custom.utils.math import quantize_obs


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        if observation.dtype == np.uint8:
            low = np.full(observation.shape, 0, dtype=np.uint8)
            high = np.full(observation.shape, 255, dtype=np.uint8)
        else:
            low = np.full(observation.shape, -float(np.inf), dtype=np.float32)
            high = np.full(observation.shape, float(np.inf), dtype=np.float32)

        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


# class GymWrapperBase(Wrapper, gym.Env):
#     metadata = None
#     render_mode = None
#     """
#     Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
#     found in the gym.core module

#     Args:
#         env (MujocoEnv): The environment to wrap.
#         keys (None or list of str): If provided, each observation will
#             consist of concatenated keys from the wrapped environment's
#             observation dictionary. Defaults to proprio-state and object-state.

#     Raises:
#         AssertionError: [Object observations must be enabled if no keys]
#     """

#     def __init__(
#             self,
#             env,
#             flatten_obs=True,
#             **kwargs
#             ):
#         # Run super method
#         super().__init__(env=env)
#         self.flatten_obs = flatten_obs
#         obs = self.env.reset()
#         low, high = self.env.action_spec
#         self.action_space = spaces.Box(low, high)
#         if self.flatten_obs:
#             flat_ob = self._flatten_obs(obs)
#             self.observation_space = convert_observation_to_space(flat_ob)
#         else:
#             self.observation_space = convert_observation_to_space(obs)

#     def _flatten_obs(self, obs_dict, verbose=False):
#         """
#         Filters keys of interest out and concatenate the information.

#         Args:
#             obs_dict (OrderedDict): ordered dictionary of observations
#             verbose (bool): Whether to print out to console as observation keys are processed

#         Returns:
#             np.array: observations flattened into a 1d array
#         """
#         ob_lst = []
#         for key in self.keys:
#             if key in obs_dict:
#                 if verbose:
#                     print("adding key: {}".format(key))
#                 ob_lst.append(np.array(obs_dict[key]).flatten())
#         return np.concatenate(ob_lst, dtype=np.float32)


class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(
        self,
        env,
        keys=None,
        info_keys=None,
        flatten_obs=True,
        render_mode=None,
        early_termination=False,
        from_pixels=False,
        channels_first=True,
        height=84,
        width=84,
        bit_depth=8,
    ):
        # Run super method
        super().__init__(env=env)
        try:
            # Create name for gym
            robots = "".join(
                [type(robot.robot_model).__name__ for robot in self.env.robots]
            )
            self.name = robots + "_" + type(self.env).__name__
        except AttributeError:
            self.name = type(self.env).__name__

        self.render_mode = render_mode
        self.early_termination = early_termination
        self.from_pixels = from_pixels
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.bit_depth = bit_depth
        try:
            # Get reward range
            self.reward_range = (0, self.env.reward_scale)
        except AttributeError:
            pass
        self.info_keys = info_keys
        if keys is None:
            if hasattr(env, "gym_keys"):
                keys = env.gym_keys
            else:
                keys = []
                try:
                    # Add object obs if requested
                    if self.env.use_object_obs:
                        keys += ["object-state"]
                except AttributeError as e:
                    print(e)
                try:
                    # Add image obs if requested
                    if self.env.use_camera_obs:
                        keys += [
                            f"{cam_name}_image"
                            for cam_name in self.env.camera_names
                        ]
                except AttributeError as e:
                    print(e)
                try:
                    # Iterate over all robots to add to state
                    for idx in range(len(self.env.robots)):
                        keys += ["robot{}_proprio-state".format(idx)]
                except AttributeError as e:
                    print(e)
        self.keys = keys
        if len(self.keys) == 0:
            raise ValueError(
                "Nothing to be flattened. Please specify either 'keys' or 'gym_keys'"
            )
        # Gym specific attributes
        self.env.spec = None

        self.flatten_obs = flatten_obs
        # set up observation and action spaces
        ob_dict = self.env.reset()
        self.modality_dims = {key: ob_dict[key].shape for key in self.keys}

        obs = self._get_obs(ob_dict)
        self.observation_space = convert_observation_to_space(obs)

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst, dtype=np.float32)

    def _get_info(self, ob_dict):
        info = {}
        if self.info_keys:
            try:
                for k in self.info_keys:
                    info.update({k: ob_dict[k]})
            except KeyError:
                pass
        return info

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        info = self._get_info(ob_dict)
        observations = self._get_obs(ob_dict)

        return observations, info

    def render(self, camera="sideview", cam_w=256, cam_h=256):
        if self.render_mode == "rgb_array":
            # assert camera in self.camera_names
            try:
                camera = self.env.render_camera
            except AttributeError:
                pass
            img = self.sim.render(
                camera_name=camera,
                width=cam_w,
                height=cam_h,
                depth=False,
            )
            return np.flipud(img)
        elif self.render_mode is None:
            self.env.render()
        else:
            raise NotImplementedError

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, truncated, info = self.env.step(action)
        info.update(self._get_info(ob_dict))
        observation = self._get_obs(ob_dict)
        if self.early_termination:
            try:
                terminated = info["success"] or info["terminated"]
            except KeyError:
                terminated = False
        else:
            terminated = False
        return observation, reward, terminated, truncated, info

    def _get_obs(self, ob_dict):
        if self.from_pixels:
            self.render_mode = "rgb_array"
            observation = self.render(cam_h=self.height, cam_w=self.width)
            if self.channels_first:
                observation = observation.transpose(2, 0, 1).copy()
            if self.bit_depth != 8:
                observation = quantize_obs(
                    observation,
                    self.bit_depth,
                    original_bit_depth=8,
                    add_noise=True,
                )

        else:
            if self.flatten_obs:
                observation = self._flatten_obs(ob_dict)
            else:
                observation = ob_dict
        return observation

    def close(self):
        pass
