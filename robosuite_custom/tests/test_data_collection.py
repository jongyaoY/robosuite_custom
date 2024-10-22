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

import argparse

import numpy as np
import tqdm
from gymnasium import spaces
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R

import robosuite_custom as suite
from robosuite_custom.wrappers import DataCollectionWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", default=0, type=int, help="env id")
parser.add_argument(
    "--num_steps", default=500, type=int, help="number of steps to collect"
)
parser.add_argument("--out_dir", type=str, help="output dir")
opt = parser.parse_args()


def controller(obs):
    action = np.zeros(6)
    socket_pose = obs["socket_pose"]
    ori_socket = R.from_rotvec(socket_pose[3:])
    ori_offset = R.from_euler("xzy", [np.pi, -np.pi / 2.0, 0])
    goal_ori = ori_socket * ori_offset
    delta_pos = obs["robot0_delta_pos"]
    eef_pose = np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_rot"]))
    delta_rot = goal_ori * R.from_rotvec(eef_pose[3:]).inv()

    has_contact = obs["robot0_contact"].sum()
    delta_pos_in_plug = R.from_rotvec(eef_pose[3:]).inv().apply(delta_pos)
    if np.linalg.norm(delta_pos_in_plug[:2]) > 0.002:
        action[:3] = np.concatenate([delta_pos_in_plug[:2], [0]])
        action[:3] = R.from_rotvec(eef_pose[3:]).apply(action[:3])
        action[:3] *= 2
    if np.linalg.norm(delta_rot.as_rotvec()) > 0.01:
        action[3:] = delta_rot.as_rotvec()

    if (
        np.linalg.norm(delta_rot.as_rotvec()) < 0.01
        and np.linalg.norm(delta_pos_in_plug[:2]) < 0.002
    ):

        if has_contact:
            action[:2] = 0.01 * np.random.uniform(-np.ones(2), np.ones(2))
            action[2] = 1
        else:
            action[2] = 0.2
        action[:3] = R.from_rotvec(eef_pose[3:]).apply(action[:3])
    return action


def collect_data(env, timesteps, controller=None):
    steps_collected = 0
    pbar = tqdm.tqdm(total=timesteps, desc="steps collected")
    low, high = env.action_spec
    action_space = spaces.Box(low, high)
    while steps_collected < timesteps:
        obs = env.reset()
        while True:
            if controller is not None:
                action = controller(obs)
            else:
                action = action_space.sample()
            obs, _, done, _ = env.step(action)
            steps_collected += 1
            pbar.update(1)
            if done:
                break
    pbar.close()
    print(f"sucessfully collected {steps_collected} steps")


if __name__ == "__main__":
    # change renderer config
    config = load_controller_config(default_controller="OSC_POSE")
    config["kp"] = 50.0
    env = suite.make(
        env_name="PegInHoleEnv",  # try with other tasks like "Stack" and "Door"
        horizon=100,
        camera_names=["agentview"],
        robots="UR10e",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
    )
    env = DataCollectionWrapper(
        env=env, directory=opt.out_dir, env_id=opt.env_id
    )
    collect_data(
        env=env, timesteps=int(opt.num_steps / 2), controller=controller
    )  # expert policy
    collect_data(
        env=env, timesteps=int(opt.num_steps / 2), controller=None
    )  # random policy
