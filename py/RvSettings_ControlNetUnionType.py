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

# From the model page: canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6)
# https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro

UNION_CONTROLNET_TYPES = {
    "canny/lineart/anime_lineart/mlsd": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "openpose": 4,
    "gray": 5,
    "low quality": 6,
}

class RvSettings_ControlNetUnionType:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET", {"tooltip": "ControlNet input object."}),
                "type": (list(UNION_CONTROLNET_TYPES.keys()), {"tooltip": "Select the ControlNet union type."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "set_controlnet_type"

    def set_controlnet_type(self, control_net, type) -> tuple:
        # Sets the control_type extra argument for the ControlNet input.
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])
        return (control_net,)

NODE_NAME = 'ControlNet Set Union Types (Flux) [RvTools]'
NODE_DESC = 'ControlNet Set Union Types (Flux)'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvSettings_ControlNetUnionType
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}