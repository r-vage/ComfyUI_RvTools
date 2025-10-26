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
from typing import List, Dict, Any, Tuple

class RvSettings_LoadDirectorySettings:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "Directory": ("STRING", {"default": "", "tooltip": "Directory path to load files from."}),
                "start_index": ("INT", {"default": 0, "min": 0, "control_after_generate": True, "tooltip": "Start index for loading files."}),
                "loadcap": ("INT", {"default": 20, "tooltip": "Maximum number of files to load."}),
            },
        }

    def execute(
        self,
        Directory: str,
        start_index: int,
        loadcap: int
    ) -> Tuple[Dict[str, object]]:
        # Return directory settings as a dict-style pipe for downstream nodes.
        pipe = {
            "directory": str(Directory),
            "start_index": int(start_index),
            "load_cap": int(loadcap),
        }
        return (pipe,)

NODE_NAME = 'Load Directory Settings [RvTools]'
NODE_DESC = 'Load Directory Settings'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_LoadDirectorySettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}