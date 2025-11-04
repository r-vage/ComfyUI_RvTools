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

from ..core import CATEGORY
from ..core import AnyType
from typing import Any, Dict, Tuple

any_type = AnyType("*")

class RvRouter_Any_DualSwitch:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.ROUTER.value
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2, "tooltip": "Select which input to output (1 or 2)."}),
            },
            "optional": {
                "input1": (any_type, {"forceInput": True, "tooltip": "First input (any type)."}),
                "input2": (any_type, {"forceInput": True, "tooltip": "Second input (any type)."}),
            }
        }

    def execute(self, Input: int, input1: Any = None, input2: Any = None) -> Tuple[Any]:
        if Input == 1:
            return (input1,)
        else:
            return (input2,)

NODE_NAME = 'Any Dual-Switch [Eclipse]'
NODE_DESC = 'Any Dual-Switch'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvRouter_Any_DualSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
