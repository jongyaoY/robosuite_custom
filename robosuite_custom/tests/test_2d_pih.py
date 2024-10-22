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

import numpy as np

import robosuite_custom as suite

if __name__ == "__main__":
    env = suite.make(
        env_name="PlanePegInHoleEnv",  # try with other tasks like "Stack" and "Door"
        render_camera="birdview",
        horizon=500,
        has_renderer=True,
        has_offscreen_renderer=True,
        control_freq=20,
    )
    env = suite.GymWrapper(
        env,
        from_pixels=False,
        channels_first=False,
        bit_depth=8,
        keys=["ft_continuous"],
        info_keys=["pos"],
    )
    # keys=["force", "torque", "velocity"])
    for _ in range(10):
        obs, _ = env.reset()
        for _ in range(100):
            action = np.random.randn(env.action_dim)
            # action = np.zeros(2)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if truncated:
                break
