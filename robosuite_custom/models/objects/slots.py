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

import os

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

from robosuite_custom.models import assets_root


class BlockSlot(MujocoXMLObject):
    def __init__(self, name="block_slot"):
        super().__init__(
            os.path.join(assets_root, "custom_objects/block_slot.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.target_site = self.worldbody.find(
            "./body/site[@name='{}target_site']".format(self.naming_prefix)
        )
        self.bottom_site = self.worldbody.find(
            "./body/site[@name='{}bottom_site']".format(self.naming_prefix)
        )
        self._obj.append(self.target_site)
        self._obj.append(self.bottom_site)

    def set_pose(self, pos, rot=None):
        self._obj.set("pos", array_to_string(pos))
        self.init_pos = pos
