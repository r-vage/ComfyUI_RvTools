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

from ..core import CATEGORY, SAMPLERS_COMFY, SCHEDULERS_ANY
from typing import Any, Dict, List, Tuple

class RvSettings_Sampler_Selection:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sampler_name": (SAMPLERS_COMFY, {"tooltip": "Select the sampler algorithm."}),
                "scheduler": (SCHEDULERS_ANY, {"tooltip": "Select the scheduler algorithm."}),
            },
        }

    def execute(self, sampler_name: Any, scheduler: Any) -> Tuple[Dict[str, Any]]:
        pipe = {
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        }
        return (pipe,)

NODE_NAME = 'Sampler Selection [RvTools]'
NODE_DESC = 'Sampler Selection'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_Sampler_Selection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
