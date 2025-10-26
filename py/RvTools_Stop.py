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

import nodes
from ..core import CATEGORY, AnyType

any = AnyType("*")

class RvTools_Stop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any,)

    FUNCTION = "execute"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def execute(self, input):
        out = input;
        nodes.interrupt_processing();
        return (out,)


NODE_NAME = 'Stop [RvTools]'
NODE_DESC = 'Stop'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvTools_Stop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}