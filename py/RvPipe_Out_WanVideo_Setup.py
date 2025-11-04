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
from ..core import CATEGORY

class RvPipe_Out_WanVideo_Setup:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input dict-style pipe containing steps, cfg, model_shift, steps_start, and steps_stop."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("steps", "cfg", "model_shift", "steps_start", "steps_stop")
    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None) -> tuple:
        # Only accept dict-style pipes now.
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_WvW_Setup expects dict-style pipes only.")

        try:
            steps_val = pipe.get("steps")
            steps = int(steps_val) if steps_val is not None else 0
        except Exception:
            steps = 0
        try:
            cfg_val = pipe.get("cfg")
            cfg = float(cfg_val) if cfg_val is not None else 0.0
        except Exception:
            cfg = 0.0
        try:
            model_shift_val = pipe.get("model_shift")
            model_shift = float(model_shift_val) if model_shift_val is not None else 0.0
        except Exception:
            model_shift = 0.0
        try:
            steps_start_val = pipe.get("steps_start")
            steps_start = int(steps_start_val) if steps_start_val is not None else 0
        except Exception:
            steps_start = 0
        try:
            steps_stop_val = pipe.get("steps_stop")
            steps_stop = int(steps_stop_val) if steps_stop_val is not None else 0
        except Exception:
            steps_stop = 0

        return (steps, cfg, model_shift, steps_start, steps_stop)

NODE_NAME = 'Pipe Out WanVideo Setup [Eclipse]'
NODE_DESC = 'Pipe Out WanVideo Setup'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_WanVideo_Setup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}