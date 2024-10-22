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

# import robosuite.utils.binding_utils
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def qpos_from_site_pose(
    sim,
    site_name,
    goal_pose,
    max_steps=100,
    tol=1e-14,
    rot_weight=1.0,
    regularization_strength=3e-2,
    regularization_threshold=0.1,
    max_update_norm=2.0,
    progress_thresh=20.0,
    inplace=False,
):
    if not inplace:
        data = copy.deepcopy(sim.data)  # TODO
        model = copy.deepcopy(sim.model)
    else:
        data = sim.data
        model = sim.model
    dtype = data.qpos.dtype

    err = np.empty(6, dtype=dtype)
    jac = np.empty((6, model.nv), dtype=dtype)  # Jacobian
    err_pos, err_rot = err[:3], err[3:]
    jac_pos, jac_rot = jac[:3], jac[3:]

    mujoco.mj_fwdPosition(model._model, data._data)
    site_id = model.site_name2id(site_name)

    dof_indices = model.dof_jntid
    update_nv = np.zeros(model.nv)

    target_pos = goal_pose[:3]
    target_rot = R.from_rotvec(goal_pose[3:])

    steps = 0
    success = False
    for _ in range(max_steps):
        site_xpos = data.get_site_xpos(site_name)
        site_xmat = data.get_site_xmat(site_name)

        site_xrot = R.from_matrix(site_xmat)
        err_norm = 0.0
        err_pos[:] = target_pos - site_xpos
        err_rot[:] = (target_rot * site_xrot.inv()).as_rotvec()
        err_norm += np.linalg.norm(err_rot) * rot_weight
        if err_norm < tol:
            success = True
            break
        else:
            # Get jacobian
            mujoco.mj_jacSite(
                model._model, data._data, jac_pos, jac_rot, site_id
            )
            jac_joints = jac[:, dof_indices]
            reg_strength = (
                regularization_strength
                if err_norm > regularization_threshold
                else 0.0
            )
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=reg_strength
            )
            update_norm = np.linalg.norm(update_joints)
            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                print(
                    "Step %2i: err_norm / update_norm (%3g) > "
                    "tolerance (%3g). Halting due to insufficient progress",
                    steps,
                    progress_criterion,
                    progress_thresh,
                )
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm
            # Write the entries for the specified joints into the full `update_nv`
            # vector.
            update_nv[dof_indices] = update_joints
            # Update `physics.qpos`, taking quaternions into account.
            mujoco.mj_integratePos(model._model, data.qpos, update_nv, 1)

            # Compute the new Cartesian position of the site.
            mujoco.mj_fwdPosition(model._model, data._data)

            steps += 1
    if not success:
        print(
            "Failed to converge after %i steps: err_norm=%3g", steps, err_norm
        )
    if not inplace:
        qpos = data.qpos.copy()
    else:
        qpos = data.qpos
    return qpos


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

    Args:
        jac_joints: The Jacobian of the end effector with respect to the joints. A
        numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
        and `nv` is the number of degrees of freedom.
        delta: The desired end-effector delta. A numpy array of shape `(3,)` or
        `(6,)` containing either position deltas, rotation deltas, or both.
        regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements. Default is zero, i.e. no regularization.

    Returns:
        An `(nv,)` numpy array of joint velocities.

    Reference:
        Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
        transpose, pseudoinverse and damped least squares methods.
        https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
