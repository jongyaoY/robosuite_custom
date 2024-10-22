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

# Calculate motion field
# https://arxiv.org/pdf/1512.02134.pdf

import math

import numpy as np
import open3d as o3d
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.transform_utils import quat2mat


def depth_to_meters(depth: np.ndarray, sim: MjSim):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    meters = near / (1 - depth * (1 - near / far))
    return meters


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def get_point_vel(
    body_linear_vel: np.ndarray,
    body_angular_vel: np.ndarray,
    body_pos: np.ndarray,
    target_pos: np.ndarray,
):
    return body_linear_vel + np.cross(body_angular_vel, target_pos - body_pos)


def vel_wrt_cam(
    vel_map: np.ndarray,
    cam_linear_vel: np.ndarray,
    cam_angular_vel: np.ndarray,
    pos_in_cam_frame: np.ndarray,
):
    return (
        vel_map - cam_linear_vel - np.cross(cam_angular_vel, pos_in_cam_frame)
    )


def get_cam_pose(sim, cam_name):
    cam_pose = np.zeros((4, 4))
    correction = quat2mat(
        [1, 0, 0, 0]
    )  # Mujoco cams have z pointing backwards
    cam_pose[:3, :3] = np.matmul(
        sim.data.get_camera_xmat(cam_name), correction
    )
    cam_pose[:3, 3] = sim.data.get_camera_xpos(cam_name)
    cam_pose[3, 3] = 1.0
    return cam_pose


def generate_point_cloud(
    depth: np.ndarray, sim: MjSim, cam_name: str, cam_w, cam_h
):
    cam_id = sim.model.camera_name2id(cam_name)
    fovy = math.radians(sim.model.cam_fovy[cam_id])
    f = 0.5 * cam_h / math.tan(fovy / 2.0)
    cam_mat = np.array(
        ((f, 0, cam_w / 2), (0, f, cam_h / 2), (0, 0, 1)), dtype=float
    )
    od_cammat = cammat2o3d(cam_mat, cam_w, cam_h)
    depth = depth_to_meters(depth, sim)
    od_depth = o3d.geometry.Image(depth)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        od_depth, od_cammat, depth_scale=1.0
    )

    cam_pose = get_cam_pose(sim, cam_name)

    o3d_cloud = o3d_cloud.transform(cam_pose)
    # o3d.io.write_point_cloud(f"{cam_name}_pcl.ply", o3d_cloud)
    points = np.asarray(o3d_cloud.points).reshape((cam_h, cam_w, 3))
    return points


def transform_to_frame(vec, frame_pos, frame_mat):
    vec_ = vec.transpose([2, 0, 1])
    shape = vec_.shape
    vec_ = vec_.reshape((shape[0], shape[1] * shape[2]))
    if frame_mat is None:
        vec_ = (vec_ - np.expand_dims(frame_pos, -1)).reshape(shape)
    else:
        vec_ = (
            frame_mat.transpose() @ (vec_ - np.expand_dims(frame_pos, -1))
        ).reshape(shape)
    return vec_.transpose([1, 2, 0])


def filter_large(img, threshold):
    img = np.where(
        abs(img) > threshold, np.zeros_like(abs(img) > threshold), img
    )
    return img


def get_motion_field(
    pixel_to_xpos: np.ndarray,
    pixel_to_bodyid: np.ndarray,
    sim: MjSim,
    cam_name: str = None,
):

    body_linear_vel = np.zeros((sim.model.nbody, 3))
    body_angular_vel = np.zeros((sim.model.nbody, 3))
    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        body_linear_vel[i] = sim.data.get_body_xvelp(body_name)
        body_angular_vel[i] = sim.data.get_body_xvelr(body_name)

    body_pos = sim.data.body_xpos[pixel_to_bodyid.squeeze(axis=-1)]
    body_linear_vel = body_linear_vel[pixel_to_bodyid.squeeze(axis=-1)]
    body_angular_vel = body_angular_vel[pixel_to_bodyid.squeeze(axis=-1)]
    vel_map = get_point_vel(
        body_linear_vel, body_angular_vel, body_pos, pixel_to_xpos
    )
    if cam_name is not None:
        cam_pose = get_cam_pose(sim, cam_name)

        cam_mat = cam_pose[:3, :3]
        cam_pos = cam_pose[:3, 3]
        pos_in_cam = transform_to_frame(pixel_to_xpos, cam_pos, None)
        cam_body_name = sim.model.body_id2name(
            sim.model.cam_bodyid[sim.model.camera_name2id(cam_name)]
        )
        cam_linear_vel = sim.data.get_body_xvelp(cam_body_name)
        cam_angular_vel = sim.data.get_body_xvelr(cam_body_name)
        vel_map = vel_wrt_cam(
            vel_map,
            cam_linear_vel=cam_linear_vel,
            cam_angular_vel=cam_angular_vel,
            pos_in_cam_frame=pos_in_cam,
        )
        # express in cam frame
        vel_in_cam = vel_map.transpose([2, 0, 1])
        shape = vel_in_cam.shape
        vel_in_cam = vel_in_cam.reshape((shape[0], shape[1] * shape[2]))
        vel_in_cam = (cam_mat.transpose() @ vel_in_cam).reshape(shape)
        vel_map = vel_in_cam.transpose([1, 2, 0])

    return filter_large(vel_map, 1.0)


def project_to_cam(pos_map, vel_map, f):
    Z = np.expand_dims(pos_map[..., -1], -1)
    p_in_cam = f * pos_map[..., :2] / Z
    vel_in_cam = (f / Z) * vel_map[..., :2] - (
        np.expand_dims(vel_map[..., -1], -1) / Z
    ) * p_in_cam
    return filter_large(vel_in_cam, 1.0)
