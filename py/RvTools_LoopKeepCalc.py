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

from typing import Tuple
from ..core import CATEGORY, cstr

class Eclipse_LoopKeepCalc:
#     Calculates frames to keep based on context length and total frames
    def __init__(self):
        pass
   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 16, "min": 1, "max": 10000, "step": 1, "tooltip": "Total number of frames in the video."}),
                "context_length": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1, "tooltip": "Context length for frame calculation."}),
                "overlap_frames": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1, "tooltip": "Number of overlapping frames between contexts."}),
                "image_loop_count": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "tooltip": "Current loop count for image processing."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames_to_keep",)
    FUNCTION = "calculate"

    def calculate(self, total_frames: int, context_length: int, overlap_frames: int, image_loop_count: int) -> Tuple[int]:
        # Calculates the number of frames to keep based on context length, overlap, and loop count for video workflows.
        # Handles None and empty input robustly.
        for name, val, default in [
            ("total_frames", total_frames, 16),
            ("context_length", context_length, 8),
            ("overlap_frames", overlap_frames, 4),
            ("image_loop_count", image_loop_count, 1)
        ]:
            if not isinstance(val, int):
                locals()[name] = default

        try:
            effective_stride = max(1, context_length - overlap_frames)
            remaining_frames = max(0, total_frames - image_loop_count)
            frames_to_keep = min(effective_stride, remaining_frames)
            return (max(0, frames_to_keep),)
        except Exception as e:
            cstr(f"Frame calculation failed: {str(e)}").error.print()
            return (0,)

NODE_NAME = 'Keep Calculator [Eclipse]'
NODE_DESC = 'Keep Calculator'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: Eclipse_LoopKeepCalc
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}