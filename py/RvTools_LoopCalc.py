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

import math
from typing import Tuple
import torch

from ..core import CATEGORY, cstr

class RvTools_LoopCalc:
#     Calculates required number of loops for processing frames with overlap
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 16, "min": 1, "max": 10000, "step": 1, "tooltip": "Total number of frames in the video."}),
                "context_length": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1, "tooltip": "Context length for frame calculation."}),
                "overlap_frames": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1, "tooltip": "Number of overlapping frames between contexts."}),
                "images": ("IMAGE", {"tooltip": "Batch of images to process."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_loops",)
    
    FUNCTION = "calculate"

    def calculate(self, total_frames: int, context_length: int, overlap_frames: int, images) -> Tuple[int]:
        # Calculates the required number of loops for processing frames with overlap in video workflows.
        # Handles None and empty input robustly.
        for name, val, default in [
            ("total_frames", total_frames, 16),
            ("context_length", context_length, 8),
            ("overlap_frames", overlap_frames, 4)
        ]:
            if not isinstance(val, int):
                locals()[name] = default

        try:
            image_count = 0
            if isinstance(images, torch.Tensor) and images.ndim > 0:
                image_count = int(images.shape[0])

            remaining_frames = max(0, total_frames - image_count)
            effective_stride = max(1, context_length - overlap_frames)
            total_loops = math.ceil(remaining_frames / effective_stride) if remaining_frames > 0 else 0
            result = max(1, int(total_loops))
            return (result,)
        except Exception as e:
            cstr(f"Loop calculation failed: {str(e)}").error.print()
            return (1,)

NODE_NAME = 'Loop Calculator [RvTools]'
NODE_DESC = 'Loop Calculator'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvTools_LoopCalc
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}