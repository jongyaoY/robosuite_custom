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

import collections
from copy import copy

import numpy as np
from robosuite.utils import RandomizationError
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import quat2mat
from scipy.spatial.transform import Rotation as R


class UniformRandomSamplerInLocalFrame(UniformRandomSampler):
    def __init__(
        self,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis=None,
        name="SocketPoseSampler",
        reference_pos=(0, 0, 0),
    ):
        if rotation_axis is not None:
            assert rotation_axis in [
                "x",
                "y",
                "z",
                "xy",
            ], "rotation axis: 'x', 'y', 'z', 'xy"
        super().__init__(
            name,
            x_range=x_range,
            y_range=y_range,
            rotation=rotation,
            rotation_axis=rotation_axis,
            reference_pos=reference_pos,
        )

    def _sample_rot(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled rotation

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation
        if self.rotation_axis == "xy":
            rot_axis = np.random.uniform(-np.ones(2), np.ones(2))
            if np.linalg.norm(rot_axis) != 0:
                rot_axis /= np.linalg.norm(rot_axis)
            noise = rot_angle * rot_axis
            noise = np.concatenate([noise, [0]])
        elif self.rotation_axis == "x":
            noise = rot_angle * np.random.uniform(-1, 1)
            noise = np.array([noise, 0, 0])
        elif self.rotation_axis == "y":
            noise = rot_angle * np.random.uniform(-1, 1)
            noise = np.array([0, noise, 0])
        elif self.rotation_axis == "z":
            noise = rot_angle * np.random.uniform(-1, 1)
            noise = np.array([0, 0, noise])
        elif self.rotation_axis is None:
            noise = np.zeros(3)
        else:
            raise RandomizationError("illegal rotation axis")
        # return R.from_euler("XYZ", noise)
        return R.from_rotvec(noise)

    def sample(self, fixtures=None, reference=None):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.


        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = 0.0
            object_x = self._sample_x(horizontal_radius)
            object_y = self._sample_y(horizontal_radius)
            object_z = self.z_offset

            delta_rot = self._sample_rot()

            # multiply this quat by the object's initial rotation if it has the attribute specified
            if hasattr(obj, "init_quat"):
                ref_rot = R.from_quat(obj.init_quat)
                # new_rot = delta_rot * ref_rot
                # quat = new_rot.as_quat()
            else:
                raise RandomizationError("init_quat of the object not set")
            # location is valid, put the object down
            pos = np.array([object_x, object_y, object_z])  # in local frame
            ref_mat = quat2mat(obj.init_quat)
            pos = ref_mat @ pos  # Transfrom to global frame
            pos += base_offset
            delta_rot = (
                ref_mat @ delta_rot.as_rotvec()
            )  # Transform delta rot to global frame
            new_rot = R.from_rotvec(delta_rot) * ref_rot
            quat = new_rot.as_quat()
            placed_objects[obj.name] = (pos, quat, obj)

        return placed_objects
