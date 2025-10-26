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

import sys

from ..core import CATEGORY

class RvLogic_IntegerGen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 1, "min": -sys.maxsize, "max": sys.maxsize, "step": 1, "control_after_generate": True, "tooltip": "Integer value to output or use with increment per queue."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PRIMITIVE.value
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    FUNCTION = "execute"

    def execute(self, value):
        # Outputs an integer value for logic operations or workflow branching.
        
        if not isinstance(value, int):
            value = 1
        return (int(value),)

NODE_NAME = 'Integer Generate [RvTools]'
NODE_DESC = 'Integer Generate'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLogic_IntegerGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}