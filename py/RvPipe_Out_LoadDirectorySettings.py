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

class RvPipe_Out_LoadDirectorySettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input dict-style pipe containing directory, start_index, and load_cap."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("directory", "start_index", "load_cap")
    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None) -> tuple:
        # Only accept dict-style pipes now.
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_LoadDirectorySettings expects dict-style pipes only.")

        directory = pipe.get("directory") or pipe.get("path") or ""
        try:
            start_index_val = pipe.get("start_index")
            start_index = int(start_index_val) if start_index_val is not None else 0
        except Exception:
            start_index = 0
        try:
            load_cap_val = pipe.get("load_cap")
            load_cap = int(load_cap_val) if load_cap_val is not None else 0
        except Exception:
            load_cap = 0

        return (directory, start_index, load_cap)

NODE_NAME = 'Pipe Out Load Directory Settings [Eclipse]'
NODE_DESC = 'Pipe Out Load Directory Settings'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_LoadDirectorySettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}