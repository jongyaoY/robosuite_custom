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

import cv2 as cv

# import matplotlib.pyplot as plt
import numpy as np
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R

import robosuite_custom as suite


def visualize_flow(flow, shape):
    hsv = np.zeros(shape=shape, dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv[..., 0] = ang
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


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
            action[:2] = 0.001 * np.random.uniform(-np.ones(2), np.ones(2))
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
        horizon=1000,
        camera_names=["agentview", "robot0_eye_in_hand"],
        robots="UR10e",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128 * 2,
        camera_depths=True,
        camera_segmentations="class",
    )
    method = cv.optflow.createOptFlow_SparseToDense()
    method = method.calc
    for ep_id in range(20):
        obs = env.reset()
        img_prvs = obs["agentview_image"]
        img_shape = img_prvs.shape
        img_prvs = cv.cvtColor(img_prvs, cv.COLOR_RGB2GRAY)
        action = np.zeros(6)
        # action[1] = -0.3
        for i in range(1000):
            action = controller(obs)
            # if i % 10 == 0:
            # action = -action
            obs, reward, done, info = env.step(action)
            # CV flow
            img_next = obs["agentview_image"]
            img_next = cv.cvtColor(img_next, cv.COLOR_RGB2GRAY)
            flow_cv = method(img_prvs, img_next, None)
            flow_cv = visualize_flow(flow_cv, img_shape)
            img_prvs = img_next

            agentview_img = obs["agentview_image"]
            agentview_img = cv.cvtColor(agentview_img, cv.COLOR_RGB2BGR)
            agentview_flow = visualize_flow(
                obs["agentview_optflow"], agentview_img.shape
            )
            hand_img = obs["robot0_eye_in_hand_image"]
            hand_img = cv.cvtColor(hand_img, cv.COLOR_RGB2BGR)
            hand_flow = visualize_flow(
                obs["robot0_eye_in_hand_optflow"], hand_img.shape
            )
            img_left = np.concatenate([hand_img, agentview_img], axis=0)
            img_middle = np.concatenate([hand_flow, agentview_flow], axis=0)
            img_right = np.concatenate([flow_cv, flow_cv], axis=0)
            img = np.concatenate([img_left, img_middle, img_right], axis=1)
            cv.imshow("flow", img)
            cv.waitKey(1)
            if done or info["success"]:
                break
    env.close()
