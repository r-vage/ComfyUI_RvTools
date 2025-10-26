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

from typing import Optional, Any
from ..core import CATEGORY, AnyType

any_type = AnyType("*")

class RvPipe_Out_Sampler_Settings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input pipe containing sampler settings."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    RETURN_TYPES = ("INT",   "FLOAT", any_type,      any_type,    "FLOAT",    "FLOAT",   "FLOAT",          "FLOAT",          "INT")
    RETURN_NAMES = ("steps", "cfg",  "sampler_name", "scheduler", "guidance", "denoise", "sigmas_denoise", "noise_strength", "seed")
    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None) -> tuple:
        """
        Extract sampler settings from pipe.
        Returns None for any values that don't exist in the pipe.
        """
        if pipe is None:
            # Return None for all outputs if pipe is None
            return (None, None, None, None, None, None, None, None, None)
        
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_Sampler_Settings expects dict-style pipes only.")

        # Extract values with fallback to None
        sampler = pipe.get("sampler_name")
        scheduler = pipe.get("scheduler") 
        steps = pipe.get("steps")
        cfg = pipe.get("cfg")
        guidance = pipe.get("guidance")
        denoise = pipe.get("denoise", 1.0)  # Default to 1.0 if not in pipe
        sigmas_denoise = pipe.get("sigmas_denoise")
        noise_strength = pipe.get("noise_strength")
        seed = pipe.get("seed")

        return (steps, cfg, sampler, scheduler, guidance, denoise, sigmas_denoise, noise_strength, seed)


NODE_NAME = "Pipe Out Sampler Settings [RvTools]"
NODE_DESC = "Pipe Out Sampler Settings"

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_Sampler_Settings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
