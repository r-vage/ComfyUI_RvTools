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

class RvLogic_Float:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 1.00,
                    "min": -sys.float_info.max,
                    "max": sys.float_info.max,
                    "step": 0.01,
                    "tooltip": "Float value to output."
                }),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PRIMITIVE.value
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)

    FUNCTION = "execute"

    def execute(self, value):
        # Outputs a float value for logic operations or workflow branching.
        
        if not isinstance(value, (float, int)):
            value = 1.0
        return (float(value),)

NODE_NAME = 'Float [Eclipse]'
NODE_DESC = 'Float'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLogic_Float
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}