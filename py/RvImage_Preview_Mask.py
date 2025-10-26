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
import sys
import random
import numpy as np
import folder_paths

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from ..core import CATEGORY

class RvImage_Preview_Mask():
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Batch of masks to preview"}),
                "Show_Masks": ("INT", {"default": -1, "min": -1, "max": sys.maxsize, "step": 1, "tooltip": "Number of masks to preview (-1 for all, 0 for none)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.IMAGE.value

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, masks, filename_prefix="ComfyUI", Show_Masks=0, prompt=None, extra_pnginfo=None):
        # Saves and previews masks from a batch, with options to limit the number of masks shown.
        # Handles None and empty input robustly.
        # Converts masks to grayscale images for visualization.
        #
        # Type safety: ensure valid masks and parameters
        if masks is None or not hasattr(masks, '__iter__') or len(masks) == 0:
            return {"ui": {"images": []}, "result": (masks,)}
        if not isinstance(filename_prefix, str) or not filename_prefix:
            filename_prefix = "ComfyUI"
        if not isinstance(Show_Masks, int):
            Show_Masks = -1

        filename_prefix += self.prefix_append
        
        # Get dimensions from first mask
        # Masks can be 2D [H, W], 3D [1, H, W], or from a list of masks
        first_mask = masks[0]
        if hasattr(first_mask, 'shape'):
            # Squeeze to get actual H, W dimensions
            temp_mask = first_mask
            while temp_mask.ndim > 2:
                temp_mask = temp_mask.squeeze(0) if temp_mask.shape[0] == 1 else temp_mask.squeeze()
            if temp_mask.ndim >= 2:
                height = temp_mask.shape[-2]
                width = temp_mask.shape[-1]
            else:
                height, width = 512, 512  # fallback
        else:
            height, width = 512, 512  # fallback
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, width, height)
        results = []

        for batch_number, mask in enumerate(masks):
            if Show_Masks == 0:
                break  # no preview for whatever reason

            # Convert mask to numpy array
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            
            # Ensure 2D mask (remove extra dimensions)
            # Handles both batch tensors [N,H,W] and lists of [1,H,W]
            while mask_np.ndim > 2:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np.squeeze(0)
                else:
                    mask_np = mask_np.squeeze()
            
            # Convert to 0-255 range for visualization
            mask_normalized = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
            
            # Create grayscale image from mask
            img = Image.fromarray(mask_normalized, mode='L')
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

            if Show_Masks > -1 and (batch_number + 1) == Show_Masks:
                break

            counter += 1

        return {"ui": {"images": results}, "result": (masks,)}

NODE_NAME = 'Preview Mask [RvTools]'
NODE_DESC = 'Preview Mask'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvImage_Preview_Mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
