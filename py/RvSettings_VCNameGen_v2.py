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

import os
import folder_paths
from ..core import CATEGORY
from typing import List, Dict, Any, Tuple

class RvSettings_VCNameGen_v2:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
#         Always execute to ensure fresh processing.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip": "Directory path to your video files."}),
                "filename_prefix": ("STRING", {"default": "vc", "tooltip": "Prefix for generated filenames."}),
                "filename_suffix_start": ("INT", {"default": 1, "min": 1, "tooltip": "Start index for filename suffix."}),
                "filename_suffix_end": ("INT", {"default": 5, "min": 1, "tooltip": "End index for filename suffix."}),
                "join_suffix_start": ("INT", {"default": 1, "min": 1, "tooltip": "Start index for join filename suffix."}),
                "simple_combine": ("BOOLEAN", {"default": False, "tooltip": "Enable simple combine mode."}),
                "file_extension": ("STRING", {"default": ".mp4", "tooltip": "File extension for generated files."}),
                "frame_load_cap": ("INT", {"default": 81, "tooltip": "Maximum number of frames to load."}),
            },
        }

    def execute(
        self,
        path: str,
        filename_prefix: str,
        filename_suffix_start: int,
        filename_suffix_end: int,
        join_suffix_start: int,
        simple_combine: bool,
        file_extension: str,
        frame_load_cap: int
    ) -> Tuple:
        # Generate lists of filenames for file and join operations.
        if not path or not isinstance(path, str):
            raise ValueError("Path is missing. Enter the Path to your Video Files.")

        # Build canonical dict-style pipe
        # File list generation
        flist: List[str] = []
        for counter in range(filename_suffix_start, filename_suffix_end + 1):
            number = str(counter)
            filename = os.path.join(path, f"{filename_prefix}_{number.zfill(5)}{file_extension}")
            flist.append(filename)
        fDict: Dict[str, List[str]] = {"FILE": flist}

        # Join file list generation
        jlist: List[str] = []
        join_end_idx = join_suffix_start + len(flist)
        for counter in range(join_suffix_start, join_end_idx):
            number = str(counter)
            filename = os.path.join(path, f"{filename_prefix}_join_{number.zfill(5)}{file_extension}")
            jlist.append(filename)
        jDict: Dict[str, List[str]] = {"JOIN": jlist}

        pipe: Dict[str, Any] = {
            "path": path,
            "frame_load_cap": frame_load_cap,
            "simple_combine": simple_combine,
            "file_dict": fDict,
            "join_dict": jDict,
        }

        return (pipe,)

NODE_NAME = 'VC-Filename Generator II [Eclipse]'
NODE_DESC = 'VC-Filename Generator II'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_VCNameGen_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}