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

any = AnyType("*")

class RvRouter_Any_Passer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any, {"tooltip": "Any input to be passed through."}),
                "Purge_VRAM": ("BOOLEAN", {"default": False, "tooltip": "If enabled, purges VRAM and unloads all models before passing latent."}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.ROUTER.value
    RETURN_TYPES = (any,)
    FUNCTION = "passthrough"

    def passthrough(self, input: object, Purge_VRAM: bool) -> tuple:
        if Purge_VRAM:
            purge_vram()

        return (input,)

NODE_NAME = 'Any Passer [RvTools]'
NODE_DESC = 'Any Passer'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvRouter_Any_Passer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}