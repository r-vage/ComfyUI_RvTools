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

import folder_paths
from ..core import CATEGORY

# Created for seamless_join_video_clips & combine_video_clips
# v1 is used for combine only; it automatically sets the 2nd filename (filename_suffix_start +1), and provides mask settings

class RvSettings_VCNameGen_v1:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always execute to ensure fresh processing

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip": "Base path for the output files."}),
                "filename_prefix": ("STRING", {"default": "vc", "tooltip": "Prefix for the filename."}),
                "filename_suffix_start": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Starting number for the filename suffix."}),
                "file_extension": ("STRING", {"default": ".mp4", "tooltip": "File extension for the output files."}),
                "frame_load_cap": ("INT", {"default": 81, "tooltip": "Maximum number of frames to load."}),
                "mask_first_frames": ("INT", {"default": 10, "tooltip": "Number of frames to mask at the start."}),
                "mask_last_frames": ("INT", {"default": 0, "tooltip": "Number of frames to mask at the end."}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    def execute(self,
                path: str,
                filename_prefix: str,
                filename_suffix_start: int,
                file_extension: str,
                frame_load_cap: int,
                mask_first_frames: int,
                mask_last_frames: int) -> tuple:
        # Generates two filenames for video clips and provides mask settings.

        if not path:
            raise ValueError("Path is missing. Enter the Path to your Video Files.")
        # Build canonical dict-style pipe
        counter = filename_suffix_start
        fDict = {}
        flist = []

        for _ in range(filename_suffix_start, filename_suffix_start + 2):
            number = str(counter)
            Filename = f"{path}\\{filename_prefix}_{number.zfill(5)}{file_extension}"
            flist.append(Filename)
            counter += 1

        fDict["FILE"] = flist

        pipe = {
            "path": path,
            "frame_load_cap": frame_load_cap,
            "mask_first_frames": mask_first_frames,
            "mask_last_frames": mask_last_frames,
            "file_dict": fDict,
        }

        return (pipe,)

NODE_NAME = 'VC-Filename Generator I [Eclipse]'
NODE_DESC = 'VC-Filename Generator I'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvSettings_VCNameGen_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}