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
import torchvision.transforms.v2 as T
import comfy.utils

from ..core import CATEGORY, AnyType

any_type = AnyType("*")

# Helper functions for image conversion
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def p(image):
    return image.permute([0,3,1,2])

def pb(image):
    return image.permute([0,2,3,1])

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


class RvConversion_ConvertToBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type, {"forceInput": True, "tooltip": "Any value to convert"}),
                "convert_to": (["IMAGE_LIST_TO_BATCH", "MASK_LIST_TO_BATCH", "LATENT_LIST_TO_BATCH"], 
                              {"default": "IMAGE_LIST_TO_BATCH", "tooltip": "Target type for conversion"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    INPUT_IS_LIST = True

    def execute(self, input, convert_to):
        """
        Convert input list to batch for image, mask, and latent operations.
        For primitive conversions (STRING, INT, FLOAT, COMBO), use ConvertPrimitive node instead.
        For image color conversions (RGB, RGBA, GRAYSCALE), use ImageConvert node instead.
        
        Note: With INPUT_IS_LIST=True, all parameters become lists.
        """
        # Extract convert_to from list (widgets are passed as lists when INPUT_IS_LIST=True)
        if isinstance(convert_to, list):
            convert_to = convert_to[0]
        
        # For IMAGE_LIST_TO_BATCH and MASK_LIST_TO_BATCH, keep input as list
        if convert_to == "IMAGE_LIST_TO_BATCH":
            return self._convert_image_list_to_batch(input)
        elif convert_to == "MASK_LIST_TO_BATCH":
            return self._convert_mask_list_to_batch(input)
        elif convert_to == "LATENT_LIST_TO_BATCH":
            return self._convert_latent_list_to_batch(input)
        
        # Unknown conversion type
        print(f"[Eclipse Convert] Unknown conversion type: {convert_to}")
        return (input,)
    
    def _convert_image_list_to_batch(self, images):
        """
        Convert image list to batch tensor.
        Based on Impact Pack's ImageListToImageBatch implementation.
        Fallback: if not an image list, return as-is.
        """
        # Check if input is valid
        if images is None:
            print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Input is None, returning as-is")
            return (images,)
        
        try:
            # Already a batch tensor
            if isinstance(images, torch.Tensor) and images.ndim == 4:
                if images.shape[0] > 0:
                    print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Already a batch tensor with shape {images.shape}")
                    return (images,)
                else:
                    print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Empty batch, returning as-is")
                    return (images,)
            
            # Must be list or tuple
            if not isinstance(images, (list, tuple)):
                print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Not a list or tuple, returning as-is")
                return (images,)
            
            if len(images) == 0:
                print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Empty list, returning as-is")
                return (images,)
            
            print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Processing list of {len(images)} images")
            
            # Single image in list
            if len(images) == 1:
                return (images[0],)
            
            # Multiple images - concatenate with upscaling if needed
            image1 = images[0]
            for i, image2 in enumerate(images[1:], 1):
                # Check if sizes match (H, W, C)
                if image1.shape[1:] != image2.shape[1:]:
                    print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Upscaling image {i} from {image2.shape[1:3]} to {image1.shape[1:3]}")
                    # movedim(-1, 1) converts [B,H,W,C] to [B,C,H,W] for upscaling
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1), 
                        image1.shape[2],  # width
                        image1.shape[1],  # height
                        "lanczos", 
                        "center"
                    ).movedim(1, -1)  # Convert back to [B,H,W,C]
                
                # Concatenate along batch dimension
                image1 = torch.cat((image1, image2), dim=0)
            
            print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH: Created batch tensor with shape {image1.shape}")
            return (image1,)
        except Exception as e:
            print(f"[Eclipse Convert] IMAGE_LIST_TO_BATCH conversion failed: {e}, returning as-is")
            import traceback
            traceback.print_exc()
            return (images,)
    
    def _convert_mask_list_to_batch(self, mask):
        """
        Convert mask list to batch tensor.
        Based on Impact Pack's MaskListToMaskBatch implementation.
        Fallback: if not a mask list, return as-is.
        """
        if mask is None:
            print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Input is None, returning as-is")
            return (mask,)
        
        try:
            # Already a batch tensor
            if isinstance(mask, torch.Tensor) and mask.ndim in (3, 4):
                if mask.shape[0] > 0:
                    print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Already a batch tensor with shape {mask.shape}")
                    return (mask,)
                else:
                    print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Empty batch, returning as-is")
                    return (mask,)
            
            # Must be list or tuple
            if not isinstance(mask, (list, tuple)):
                print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Not a list or tuple, returning as-is")
                return (mask,)
            
            if len(mask) == 0:
                print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Empty list, creating default empty mask")
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
                return (empty_mask,)
            
            print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Processing list of {len(mask)} masks")
            
            # Convert all masks to 3D format [B, H, W]
            masks_3d = [make_3d_mask(m) for m in mask]
            
            # Get target shape from first mask
            target_shape = masks_3d[0].shape[1:]  # [H, W]
            print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Target shape (H,W): {target_shape}")
            
            # Upscale masks if needed to match target shape
            upscaled_masks = []
            for i, m in enumerate(masks_3d):
                if m.shape[1:] != target_shape:
                    print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Upscaling mask {i} from {m.shape[1:]} to {target_shape}")
                    # Add channel dimension for upscaling: [B,H,W] -> [B,3,H,W]
                    m = m.unsqueeze(1).repeat(1, 3, 1, 1)
                    # Upscale
                    m = comfy.utils.common_upscale(m, target_shape[1], target_shape[0], "lanczos", "center")
                    # Remove channel dimension: [B,3,H,W] -> [B,H,W]
                    m = m[:, 0, :, :]
                
                upscaled_masks.append(m)
            
            # Concatenate all masks at once along batch dimension
            result = torch.cat(upscaled_masks, dim=0)
            print(f"[Eclipse Convert] MASK_LIST_TO_BATCH: Created batch tensor with shape {result.shape}")
            return (result,)
        except Exception as e:
            print(f"[Eclipse Convert] MASK_LIST_TO_BATCH conversion failed: {e}, returning as-is")
            import traceback
            traceback.print_exc()
            return (mask,)
    
    def _convert_latent_list_to_batch(self, latents):
        """
        Convert latent list to batch tensor.
        Latents are dictionaries with structure: {"samples": torch.Tensor}
        A latent batch is a single dict where "samples" tensor has shape [B, C, H, W]
        A latent list is a list of dicts, each with "samples" of shape [1, C, H, W]
        """
        if latents is None:
            print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Input is None, returning as-is")
            return (latents,)
        
        try:
            # Check if input is already a batch dict
            if isinstance(latents, dict) and "samples" in latents:
                samples = latents["samples"]
                if isinstance(samples, torch.Tensor) and samples.ndim == 4:
                    print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Already a latent batch with shape {samples.shape}")
                    return (latents,)
            
            # Must be list or tuple
            if not isinstance(latents, (list, tuple)):
                print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Not a list or tuple, returning as-is")
                return (latents,)
            
            if len(latents) == 0:
                print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Empty list, returning as-is")
                return (latents,)
            
            print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Processing list of {len(latents)} latents")
            
            # Each item should be a dict with "samples" key
            if not all(isinstance(item, dict) and "samples" in item for item in latents):
                print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Not all items are latent dicts, returning as-is")
                return (latents,)
            
            # Single latent in list
            if len(latents) == 1:
                return (latents[0],)
            
            # Multiple latents - concatenate samples tensors
            latent1 = latents[0]
            samples1 = latent1["samples"]
            
            for i, latent2 in enumerate(latents[1:], 1):
                samples2 = latent2["samples"]
                
                # Check if shapes match (C, H, W)
                if samples1.shape[1:] != samples2.shape[1:]:
                    print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Upscaling latent {i} from {samples2.shape[1:]} to {samples1.shape[1:]}")
                    # Latents are already in [B, C, H, W] format
                    samples2 = comfy.utils.common_upscale(
                        samples2,
                        samples1.shape[3],  # width
                        samples1.shape[2],  # height
                        "lanczos",
                        "center"
                    )
                
                # Concatenate along batch dimension
                samples1 = torch.cat((samples1, samples2), dim=0)
            
            # Create result batch dict
            result = {"samples": samples1}
            print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH: Created latent batch with shape {samples1.shape}")
            return (result,)
        except Exception as e:
            print(f"[Eclipse Convert] LATENT_LIST_TO_BATCH conversion failed: {e}, returning as-is")
            import traceback
            traceback.print_exc()
            return (latents,)


NODE_NAME = "Convert To Batch [Eclipse]"
NODE_DESC = "Convert To Batch"

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_ConvertToBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
