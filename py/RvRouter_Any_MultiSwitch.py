# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from ..core import CATEGORY, purge_vram
from ..core import AnyType

any_type = AnyType("*")

class RvRouter_Any_MultiSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1, "tooltip": "Number of ANY inputs to expose. Inputs update automatically."}),
                "Purge_VRAM": ("BOOLEAN", {"default": False, "tooltip": "If enabled, purges VRAM before switching."}),
            },
            "optional": {
                "any_1": (any_type, {"tooltip": "Any input #1 (highest priority). Leave empty to bypass."}),
                "any_2": (any_type, {"tooltip": "Any input #2 (used if #1 is empty)."}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "select"
    CATEGORY = CATEGORY.MAIN.value +  CATEGORY.ROUTER.value
    DESCRIPTION = "Multi-switch for ANY inputs. Inputs update automatically when inputcount changes." 

    def select(self, inputcount, Purge_VRAM=False, **kwargs):
        if Purge_VRAM:
            purge_vram()

        def _is_empty(v):
            if v is None:
                return True
            if isinstance(v, (tuple, list)) and len(v) == 0:
                return True
            if isinstance(v, dict) and len(v) == 0:
                return True
            if isinstance(v, str) and v.strip() == "":
                return True
            return False

        for i in range(1, max(1, inputcount) + 1):
            key = f"any_{i}"
            val = kwargs.get(key)
            if not _is_empty(val):
                return (val,)

        raise RuntimeError(f"RvRouter_Any_MultiSwitch: no value found among any_1..any_{inputcount}.")

NODE_NAME = 'Any Multi-Switch [RvTools]'
NODE_DESC = 'Any Multi-Switch'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvRouter_Any_MultiSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
