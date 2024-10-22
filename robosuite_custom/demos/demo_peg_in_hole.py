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

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R

import robosuite_custom as suite


def controller(obs):
    action = np.zeros(6)
    socket_pose = obs["socket_pose"]
    ori_socket = R.from_rotvec(socket_pose[3:])
    ori_offset = R.from_euler("xzy", [np.pi, -np.pi / 2.0, 0])
    goal_ori = ori_socket * ori_offset
    eef_pose = np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_rot"]))
    delta_pos = obs["socket_pose"][:3] - eef_pose[:3]
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


if __name__ == "__main__":
    # change renderer config
    config = load_controller_config(default_controller="OSC_POSE")
    config["kp"] = 50.0
    env = suite.make(
        env_name="PegInHoleEnv",  # try with other tasks like "Stack" and "Door"
        render_camera="birdview",
        controller_configs=config,
        # render_camera="sideview",
        # render_camera="agentview", #on-screen render
        # camera_names="birdview",
        horizon=500,
        robots="UR10e",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)
    ax.set_title("forces")
    ax_2.set_title("torques")
    ax.set_ylim([-200, 200])
    buffer_size = 200
    for ep_id in range(20):
        print(f"{ep_id}")
        obs = env.reset()
        xs = deque(maxlen=buffer_size)
        fx = deque(maxlen=buffer_size)
        fy = deque(maxlen=buffer_size)
        fz = deque(maxlen=buffer_size)

        mx = deque(maxlen=buffer_size)
        my = deque(maxlen=buffer_size)
        mz = deque(maxlen=buffer_size)
        for i in range(1000):
            action = controller(obs)
            obs, reward, done, info = env.step(action)
            xs.append(i)
            fx.append(obs["robot0_force"][0])
            fy.append(obs["robot0_force"][1])
            fz.append(obs["robot0_force"][2])
            mx.append(obs["robot0_torque"][0])
            my.append(obs["robot0_torque"][1])
            mz.append(obs["robot0_torque"][2])

            if done or info["success"]:
                break
            env.render()  # render on display
            # plt.pause(0.00002)
        ax.plot(xs, fx, "r")
        ax.plot(xs, fy, "g")
        ax.plot(xs, fz, "b")
        ax_2.plot(xs, mx, "r")
        ax_2.plot(xs, my, "g")
        ax_2.plot(xs, mz, "b")
        plt.show()
        ax.clear()
        ax_2.clear()
    env.close()
