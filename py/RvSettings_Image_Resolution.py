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
from ..core.common import RESOLUTION_PRESETS, RESOLUTION_MAP
from typing import Dict, Any, Tuple

class RvSettings_Image_Resolution:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "resolution": (RESOLUTION_PRESETS[1:], {
                    "tooltip": "Select the aspect ratio and resolution for your image."
                }),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"

    def execute(self, resolution: str) -> Tuple[int, int]:
        # Return width and height for the selected resolution string.

        width, height = RESOLUTION_MAP.get(resolution, (512, 512))
        return width, height

NODE_NAME = 'Image Resolution [Eclipse]'
NODE_DESC = 'Image Resolution'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_Image_Resolution
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}