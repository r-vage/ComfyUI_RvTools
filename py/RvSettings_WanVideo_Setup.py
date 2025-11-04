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
from typing import Any, Dict, Tuple

class RvSettings_WanVideo_Setup:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "steps": ("INT", {"default": 4, "min": 1, "tooltip": "Number of steps for video processing."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "tooltip": "Classifier-Free Guidance scale."}),
                "model_shift": ("FLOAT", {"default": 5.0, "min": 0, "tooltip": "Model shift value for video batch."}),
                "steps_start": ("INT", {"default": 2, "min": -1, "tooltip": "Start index for split steps."}),
                "steps_stop": ("INT", {"default": 2, "min": -1, "max": 10000, "tooltip": "End index for split steps."}),
            },
        }

    def execute(
        self,
        steps: int,
        cfg: float,
        model_shift: float,
        steps_start: int,
        steps_stop: int
    ) -> Tuple:
        # Return a canonical dict-style pipe so downstream nodes can access fields by name
        pipe = {
            "steps": int(steps),
            "cfg": float(cfg),
            "model_shift": float(model_shift),
            "steps_start": int(steps_start),
            "steps_stop": int(steps_stop),
        }
        return (pipe,)

NODE_NAME = 'WanVideo Setup [Eclipse]'
NODE_DESC = 'WanVideo Setup'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_WanVideo_Setup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
