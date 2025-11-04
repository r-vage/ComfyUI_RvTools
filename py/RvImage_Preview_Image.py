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

class RvImage_Preview_Image():
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of images to preview"}),
                "Show_Images": ("INT", {"default": -1, "min": -1, "max": sys.maxsize, "step": 1, "tooltip": "Number of images to preview (-1 for all, 0 for none)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.IMAGE.value

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, images, filename_prefix="ComfyUI", Show_Images=0, prompt=None, extra_pnginfo=None):
        # Saves and previews images from a batch, with options to limit the number of images shown.
        # Handles None and empty input robustly.
        # Supports both batch tensors [N,H,W,C] and lists of single-image tensors [1,H,W,C].
        #
        # Type safety: ensure valid images and parameters
        if images is None or not hasattr(images, '__iter__') or len(images) == 0:
            return {"ui": {"images": []}, "result": (images,)}
        if not isinstance(filename_prefix, str) or not filename_prefix:
            filename_prefix = "ComfyUI"
        if not isinstance(Show_Images, int):
            Show_Images = -1

        # Get first image for dimensions
        first_img = images[0]
        # If first image is a single-image tensor [1,H,W,C], squeeze it for dimensions
        if hasattr(first_img, 'shape') and first_img.shape[0] == 1:
            height, width = first_img.shape[1], first_img.shape[2]
        else:
            height, width = first_img.shape[0], first_img.shape[1]

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, width, height)
        results = []

        for batch_number, image in enumerate(images):
            if Show_Images == 0:
                break  # no preview for whatever reason

            # If image is [1,H,W,C], squeeze to [H,W,C]
            if hasattr(image, 'shape') and image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

            if Show_Images > -1 and (batch_number + 1) == Show_Images:
                break

            counter += 1

        return {"ui": {"images": results}, "result": (images,)}

NODE_NAME = 'Preview Image [Eclipse]'
NODE_DESC = 'Preview Image'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvImage_Preview_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}