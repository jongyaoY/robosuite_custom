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


# ------------------------------------------------------------------------ #
# Pixel manipulation
# ------------------------------------------------------------------------ #
def quantize_obs(
    obs: np.ndarray,
    bit_depth: int,
    original_bit_depth: int = 8,
    add_noise: bool = False,
):
    """Quantizes an array of pixel observations to the desired bit depth.

    Args:
        obs (np.ndarray): the array to quantize.
        bit_depth (int): the desired bit depth.
        original_bit_depth (int, optional): the original bit depth, defaults to 8.
        add_noise (bool, optional): if ``True``, uniform noise in the range
            (0, 2 ** (8 - bit_depth)) will be added. Defaults to ``False``.`

    Returns:
        (np.ndarray): the quantized version of the array.
    """
    ratio = 2 ** (original_bit_depth - bit_depth)
    quantized_obs = (obs // ratio) * ratio
    if add_noise:
        quantized_obs = quantized_obs.astype(
            np.double
        ) + ratio * np.random.rand(*obs.shape)
    return quantized_obs
