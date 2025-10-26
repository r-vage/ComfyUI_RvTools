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
import numpy as np
from PIL import Image

from ..core import CATEGORY, AnyType

any_type = AnyType("*")

# Helper functions for image conversion
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Helper function for mask conversion
def make_3d_mask(mask):
    """Convert mask to 3D format"""
    if not hasattr(mask, "shape"):
        return mask
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


class RvConversion_ConvertToList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type, {"forceInput": True, "tooltip": "Any value to convert to list format"}),
                "convert_to": (["IMAGE_BATCH_TO_LIST", "MASK_BATCH_TO_LIST", "LATENT_BATCH_TO_LIST"], 
                              {"default": "IMAGE_BATCH_TO_LIST", "tooltip": "Target list conversion type"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self, input, convert_to: str):
        """
        Convert input to list format.
        For image color conversions (RGB, RGBA, GRAYSCALE), use ImageConvert node instead.
        Returns list-based conversions based on convert_to selection.
        """
        # Handle image conversions
        if convert_to == "IMAGE_BATCH_TO_LIST":
            return self._convert_image_batch_to_list(input)
        
        # Handle mask conversions
        elif convert_to == "MASK_BATCH_TO_LIST":
            return self._convert_mask_batch_to_list(input)
        
        # Handle latent conversions
        elif convert_to == "LATENT_BATCH_TO_LIST":
            return self._convert_latent_batch_to_list(input)
        
        # Fallback
        return ([input],)
    
    def _convert_image_batch_to_list(self, images):
        """
        Convert image batch to list of images.
        Fallback: if not an image batch, return as single-item list.
        """
        # Check if input is tensor-like
        if not isinstance(images, (torch.Tensor, list, tuple)):
            print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Input is not a batch, returning as single-item list")
            return ([images],)
        
        try:
            if isinstance(images, (list, tuple)):
                if len(images) == 0:
                    print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Empty list, returning as-is")
                    return ([],)
                # Validate each image is a tensor with shape
                for i, img in enumerate(images):
                    if not hasattr(img, "shape") or img.ndim < 3:
                        print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Image at index {i} is not a valid tensor, returning as-is")
                        return (list(images),)
                return (list(images),)
            
            if not hasattr(images, "shape") or not hasattr(images, "__getitem__"):
                print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Not a valid batch tensor, returning as single-item list")
                return ([images],)
            
            batch_size = images.shape[0]
            if batch_size == 0:
                print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Empty batch, returning empty list")
                return ([],)
            
            # Validate each image in batch and ensure 4D shape [1, H, W, C]
            img_list = []
            for i in range(batch_size):
                img = images[i:i + 1, ...]
                # Verify it's 4D tensor with batch dimension of 1
                if not hasattr(img, "shape") or img.ndim != 4:
                    print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST: Image at batch index {i} is not valid 4D tensor, returning as single-item list")
                    return ([images],)
                img_list.append(img)
            
            return (img_list,)
        except Exception as e:
            print(f"[RvTools ConvertToList] IMAGE_BATCH_TO_LIST conversion failed: {e}, returning as single-item list")
            return ([images],)
    
    def _convert_mask_batch_to_list(self, masks):
        """
        Convert mask batch to list of masks.
        Fallback: if not a mask batch, return as single-item list.
        """
        if masks is None:
            print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Input is None, returning empty list")
            return ([],)
        
        try:
            # Already a list/tuple
            if isinstance(masks, (list, tuple)):
                if len(masks) == 0:
                    print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Empty list, returning as-is")
                    return ([],)
                # Validate each mask is a tensor with shape
                for i, m in enumerate(masks):
                    if not hasattr(m, "shape") or m.ndim < 2:
                        print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Mask at index {i} is not a valid tensor, returning as-is")
                        return (list(masks),)
                return (list(masks),)
            
            # Process batch tensor
            if hasattr(masks, "shape") and hasattr(masks, "__getitem__"):
                batch_size = masks.shape[0]
                if batch_size == 0:
                    print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Empty batch, returning empty list")
                    return ([],)
                
                # Convert to list of 3D masks
                mask_list = []
                for i in range(batch_size):
                    m = make_3d_mask(masks[i])
                    if not hasattr(m, "shape") or m.ndim < 2:
                        print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Mask at batch index {i} is not valid, returning as single-item list")
                        return ([masks],)
                    mask_list.append(m)
                return (mask_list,)
            
            print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST: Not a valid mask batch, returning as single-item list")
            return ([masks],)
        except Exception as e:
            print(f"[RvTools ConvertToList] MASK_BATCH_TO_LIST conversion failed: {e}, returning as single-item list")
            return ([masks],)
    
    def _convert_latent_batch_to_list(self, latents):
        """
        Convert latent batch to list of individual latents.
        A latent batch is a dict: {"samples": torch.Tensor} with shape [B, C, H, W]
        A latent list is a list of dicts: [{"samples": tensor}, ...] each with shape [1, C, H, W]
        Fallback: if not a latent batch, return as single-item list.
        """
        if latents is None:
            print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Input is None, returning empty list")
            return ([],)
        
        try:
            # Already a list/tuple of latent dicts
            if isinstance(latents, (list, tuple)):
                if len(latents) == 0:
                    print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Empty list, returning as-is")
                    return ([],)
                # Validate each latent is a dict with "samples"
                for i, latent in enumerate(latents):
                    if not isinstance(latent, dict) or "samples" not in latent:
                        print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Item at index {i} is not a valid latent dict, returning as-is")
                        return (list(latents),)
                return (list(latents),)
            
            # Process latent batch dict
            if isinstance(latents, dict) and "samples" in latents:
                samples = latents["samples"]
                
                if not hasattr(samples, "shape") or not hasattr(samples, "__getitem__"):
                    print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: 'samples' is not a valid tensor, returning as single-item list")
                    return ([latents],)
                
                batch_size = samples.shape[0]
                if batch_size == 0:
                    print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Empty batch, returning empty list")
                    return ([],)
                
                # Convert to list of individual latent dicts
                latent_list = []
                for i in range(batch_size):
                    # Extract single latent with batch dimension: [1, C, H, W]
                    single_sample = samples[i:i + 1, ...]
                    latent_dict = {"samples": single_sample}
                    latent_list.append(latent_dict)
                
                print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Split batch into {len(latent_list)} individual latents")
                return (latent_list,)
            
            print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST: Not a valid latent batch dict, returning as single-item list")
            return ([latents],)
        except Exception as e:
            print(f"[RvTools ConvertToList] LATENT_BATCH_TO_LIST conversion failed: {e}, returning as single-item list")
            import traceback
            traceback.print_exc()
            return ([latents],)


NODE_NAME = "Convert to List [RvTools]"
NODE_DESC = "Convert to List"

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_ConvertToList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
