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
from typing import Any, Dict, Tuple

class RvSettings_Sampler_Settings_NI:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "allow_overwrite": ("BOOLEAN", {"default": True, "tooltip": "When enabled, allows direct inputs to IO nodes to overwrite this node's values."}),
                "sampler_name": (SAMPLERS_COMFY, {"tooltip": "Select the sampler algorithm."}),
                "scheduler": (SCHEDULERS_ANY, {"tooltip": "Select the scheduler algorithm."}),
                "steps": ("INT", {"default": 20, "min": 1, "step": 1, "tooltip": "Number of sampling steps."}),
                "cfg": ("FLOAT", {"default": 3.50, "min": 0, "step": 0.1, "tooltip": "Classifier-Free Guidance scale."}),
                "guidance": ("FLOAT", {"default": 3.50, "min": 0, "step": 0.1, "tooltip": "Flux guidance scale."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1, "tooltip": "Denoise strength (0-1)."}),
                "sigmas_denoise": ("FLOAT", {"default": 0.45, "min": 0, "step": 0.1, "tooltip": "Sigma denoise value."}),
                "noise_strength": ("FLOAT", {"default": 0.50, "min": 0, "step": 0.1, "tooltip": "Noise strength value."}),
            },
        }

    def execute(self, allow_overwrite: bool, sampler_name: Any, scheduler: Any, steps: int, cfg: float, guidance: float,
                denoise: float, sigmas_denoise: float, noise_strength: float) -> Tuple:
        
        pipe = {
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "steps": int(steps),
            "cfg": float(cfg),
            "guidance": float(guidance),
            "denoise": float(denoise),
            "sigmas_denoise": float(sigmas_denoise),
            "noise_strength": float(noise_strength),
            "seed": int(0),
            "_allow_overwrite": allow_overwrite,  # Flag for IO nodes
        }
        return (pipe,)

NODE_NAME = 'Sampler Settings NI [RvTools]'
NODE_DESC = 'Sampler Settings NI'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_Sampler_Settings_NI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
