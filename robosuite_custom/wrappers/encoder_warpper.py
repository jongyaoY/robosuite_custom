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

import gym
import numpy as np
import torch
from gym import spaces
from rl.sensor_fusion import ProcessForce, SensorFusionSelfSupervised, ToTensor
from robosuite.wrappers import Wrapper
from scipy.spatial.transform import Rotation as R
from torchvision import transforms


class EncoderWrapper(Wrapper, gym.Env):
    def __init__(self, env, configs):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
        """
        super().__init__(env)
        # Create name for gym
        robots = "".join(
            [type(robot.robot_model).__name__ for robot in self.env.robots]
        )
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)
        # Load model
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            action_dim=configs["action_dim"],
        ).to(self.device)
        self.transform = transforms.Compose(
            [
                ProcessForce(32, "frc_in", tanh=True),
                ToTensor(device=self.device),
            ]
        )
        ckpt = torch.load(configs["load"])
        self.model.load_state_dict(ckpt)
        self.model.eval()

        # set up observation and action spaces
        obs = self.reset()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            (np.array) state in latent space
        """
        observations = super().reset()
        return self._encode(observations)

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) state in latent space
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        observations, reward, terminated, info = super().step(action)
        return self._encode(observations), reward, terminated, info

    def _encode(self, observations):
        input = self._observations_to_tensors(observations)
        state = self.model(**input, action_in=None)[-1]
        state = np.squeeze(state.cpu().detach().numpy(), 0)
        return state

    def _observations_to_tensors(self, observations):
        pf = self.robots[0].robot_model.naming_prefix
        camera_name = self.camera_names[0] + "_"
        image = observations[camera_name + "image"]
        depth = observations[camera_name + "depth"]
        force = observations[pf + "ee_forces_continuous"]
        ee_pos = observations[pf + "eef_pos"]
        ee_ori = R.from_quat(observations[pf + "eef_quat"]).as_rotvec()
        ee_vel = observations[pf + "eef_vel"][:3]
        ee_vel_ori = observations[pf + "eef_vel"][3:]
        proprio = np.concatenate([ee_pos, ee_ori, ee_vel, ee_vel_ori])
        sample = {
            "vis_in": image,
            "depth_in": depth,
            "frc_in": force,
            "proprio_in": proprio,
        }
        sample = self.transform(sample)
        for k in sample.keys():
            sample[k] = torch.unsqueeze(sample[k], 0)
        return sample

    def render(self, mode=None, camera="sideview", cam_w=256, cam_h=256):
        if mode == "rgb_array":
            # assert camera in self.camera_names
            img = self.sim.render(
                camera_name=camera,
                width=cam_w,
                height=cam_h,
                depth=False,
            )
            return np.flipud(img)
        elif mode is None:
            self.env.render()
        else:
            raise NotImplementedError
