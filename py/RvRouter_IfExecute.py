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

from ..core import CATEGORY, purge_vram
from ..core import AnyType
from typing import Any, Dict, Tuple

any = AnyType("*")

class RvSwitch_IfExecute:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.ROUTER.value
    RETURN_TYPES = (any,)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "on_true": (any, {"tooltip": "Value to return if boolean is True."}),
                "on_false": (any, {"tooltip": "Value to return if boolean is False."}),
                "boolean": ("BOOLEAN", {"forceInput": True, "tooltip": "Condition to select on_true or on_false."}),
                "Purge_VRAM": ("BOOLEAN", {"default": False, "tooltip": "If True, purges VRAM before switching."}),
            }
        }

    def execute(self, on_true: Any, on_false: Any, boolean: bool = True, Purge_VRAM: bool = False) -> Tuple[Any]:

        if Purge_VRAM:
            purge_vram()    
            
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

NODE_NAME = 'IF A Else B [RvTools]'
NODE_DESC = 'IF A Else B'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSwitch_IfExecute
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
