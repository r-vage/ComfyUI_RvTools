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

import torch
import comfy.model_management
import folder_paths
from ..core import CATEGORY

class RvPipe_Out_SmartFolder:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input pipe from Smart Folder containing generation mode (image/video) and all relevant parameters."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    
    # Output types - provides all outputs for both image and video modes
    RETURN_TYPES = (
        "STRING",  # path: Output directory path
        "INT",     # width: Frame/image width in pixels
        "INT",     # height: Frame/image height in pixels
        "INT",     # batch_size: Number of items per batch
        "LATENT",  # latent: Generated empty latent tensor (for image mode)
        "FLOAT",   # frame_rate: Video frames per second (for video mode)
        "INT",     # frame_load_cap: Maximum frames to load (for video mode)
        "INT",     # context_length: Context length for WAN models (for video mode)
        "INT",     # overlap: Overlap frames between clips (for video mode)
        "INT",     # skip_first_frames: Number of initial frames to skip (for video mode)
        "INT",     # select_every_nth: Frame sampling rate (for video mode)
        "INT",     # seed: Random seed value
    )
    
    RETURN_NAMES = (
        "path",
        "width", 
        "height", 
        "batch_size",
        "latent",
        "frame_rate",
        "frame_load_cap",
        "context_length",
        "overlap",
        "skip_first_frames",
        "select_every_nth",
        "seed",
    )
    
    FUNCTION = "execute"

    def execute(self, pipe=None) -> tuple:
        # Validate pipe input
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_SmartFolder expects dict-style pipes only.")

        # Get path from pipe, if empty use ComfyUI's default output path
        path = pipe.get("path") or ""
        if not path:
            path = folder_paths.get_output_directory()
        
        # Get common parameters (no defaults - use None if missing)
        width = pipe.get("width")
        height = pipe.get("height")
        batch_size = pipe.get("batch_size")

        # Generate latent only if we have valid dimensions (useful for image mode)
        output_latent = None
        if width is not None and height is not None and batch_size is not None:
            try:
                output_latent = self.generate(int(width), int(height), int(batch_size))[0]
            except Exception:
                output_latent = None

        # Get video-specific parameters (None if not in pipe)
        frame_rate = pipe.get("frame_rate")
        frame_load_cap = pipe.get("frame_load_cap")
        context_length = pipe.get("context_length")
        overlap = pipe.get("overlap")
        skip_first_frames = pipe.get("skip_first_frames")
        select_every_nth = pipe.get("select_every_nth")
        
        # Get seed
        try:
            seed_val = pipe.get("seed")
            seed = int(seed_val) if seed_val is not None else None
        except Exception:
            seed = None

        return (
            path,
            width,
            height,
            batch_size,
            output_latent,
            frame_rate,
            frame_load_cap,
            context_length,
            overlap,
            skip_first_frames,
            select_every_nth,
            seed,
        )

    def generate(self, width, height, batch_size=1):
        """Generate empty latent tensor for image generation"""
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)
                

NODE_NAME = 'Pipe Out Smart Folder [Eclipse]'
NODE_DESC = 'Pipe Out Smart Folder'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_SmartFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
