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
import subprocess

from ..core import CATEGORY

# Import pilgram for style filters
try:
    import pilgram
except ImportError:
    subprocess.check_call(['pip', 'install', 'pilgram'])

def tensor2pil(image):
    # Convert tensor to PIL Image
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    # Convert PIL Image to tensor
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def is_rgb_tensor(tensor):
    # Check if tensor is RGB (3 channels)
    return tensor.shape[-1] == 3

def is_rgba_tensor(tensor):
    # Check if tensor is RGBA (4 channels)
    return tensor.shape[-1] == 4

def is_grayscale_tensor(tensor):
    # Check if tensor is grayscale (1 channel)
    return tensor.shape[-1] == 1

def convert_to_rgb(tensor):
    # Convert tensor to RGB
    if is_rgb_tensor(tensor):
        return tensor
    elif is_rgba_tensor(tensor):
        # Remove alpha channel
        return tensor[..., :3]
    elif is_grayscale_tensor(tensor):
        # Expand grayscale to RGB by repeating the channel
        return tensor.repeat(1, 1, 1, 3)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def convert_to_grayscale(tensor):
    # Convert tensor to grayscale using luminance formula.
    # Returns RGB format (3 channels) with same grayscale value in all channels
    # to maintain ComfyUI IMAGE format compatibility.
    if is_grayscale_tensor(tensor):
        # Already grayscale, convert to 3-channel format
        return tensor.repeat(1, 1, 1, 3)
    elif is_rgb_tensor(tensor) or is_rgba_tensor(tensor):
        # Use standard luminance formula: Y = 0.299R + 0.587G + 0.114B
        rgb = tensor[..., :3]  # Use only RGB channels
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=tensor.dtype, device=tensor.device)
        grayscale = torch.sum(rgb * weights, dim=-1, keepdim=True)
        # Repeat to 3 channels for ComfyUI compatibility
        return grayscale.repeat(1, 1, 1, 3)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def remove_alpha_channel(tensor):
    # Remove alpha channel from tensor
    if is_rgba_tensor(tensor):
        return tensor[..., :3]
    else:
        # No alpha channel to remove
        return tensor


class RvConversion_ImageConvert:
    # Convert images between different color spaces and formats.
    # Supports RGB and Grayscale conversions.
    # Multiple conversions can be applied in sequence.
    # Optionally apply Instagram-like style filters.

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "to_rgb": ("BOOLEAN", {"default": False, "tooltip": "Convert to RGB (3 channels)"}),
                "to_grayscale": ("BOOLEAN", {"default": False, "tooltip": "Convert to grayscale"}),
                "remove_alpha": ("BOOLEAN", {"default": False, "tooltip": "Remove alpha channel"}),
                "style": (["none", "1977", "aden", "brannan", "brooklyn", "clarendon", "earlybird", 
                          "gingham", "hudson", "inkwell", "kelvin", "lark", "lofi", "maven", "mayfair", 
                          "moon", "nashville", "perpetua", "reyes", "rise", "slumber", "stinson", 
                          "toaster", "valencia", "walden", "willow", "xpro2"], 
                         {"default": "none", "tooltip": "Instagram-like style filter to apply"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value

    def execute(self, images, to_rgb=False, to_grayscale=False, remove_alpha=False, style="none"):
        # Convert images to the specified color space and optionally apply style filter.
        # Multiple conversions are applied in the order:
        # 1. Remove alpha (if selected)
        # 2. To RGB (if selected)
        # 3. To Grayscale (if selected)
        # 4. Apply style filter (if selected)
        # 
        # Args:
        #     images: Input image tensor [B, H, W, C]
        #     to_rgb: Convert to RGB
        #     to_grayscale: Convert to grayscale
        #     remove_alpha: Remove alpha channel
        #     style: Instagram-like style filter
        # 
        # Returns:
        #     Tuple containing converted image tensor
        
        result = images
        
        # Apply conversions in sequence
        if remove_alpha:
            result = remove_alpha_channel(result)
        
        if to_rgb:
            result = convert_to_rgb(result)
        
        if to_grayscale:
            result = convert_to_grayscale(result)
        
        # Apply style filter if selected
        if style != "none":
            result = self._apply_style(result, style)
        
        return (result,)
    
    def _apply_style(self, images, style):
        # Apply Instagram-like style filter to images
        style_map = {
            "1977": pilgram._1977,
            "aden": pilgram.aden,
            "brannan": pilgram.brannan,
            "brooklyn": pilgram.brooklyn,
            "clarendon": pilgram.clarendon,
            "earlybird": pilgram.earlybird,
            "gingham": pilgram.gingham,
            "hudson": pilgram.hudson,
            "inkwell": pilgram.inkwell,
            "kelvin": pilgram.kelvin,
            "lark": pilgram.lark,
            "lofi": pilgram.lofi,
            "maven": pilgram.maven,
            "mayfair": pilgram.mayfair,
            "moon": pilgram.moon,
            "nashville": pilgram.nashville,
            "perpetua": pilgram.perpetua,
            "reyes": pilgram.reyes,
            "rise": pilgram.rise,
            "slumber": pilgram.slumber,
            "stinson": pilgram.stinson,
            "toaster": pilgram.toaster,
            "valencia": pilgram.valencia,
            "walden": pilgram.walden,
            "willow": pilgram.willow,
            "xpro2": pilgram.xpro2,
        }
        
        if style not in style_map:
            return images
        
        filter_func = style_map[style]
        tensors = []
        
        for img in images:
            styled_img = pil2tensor(filter_func(tensor2pil(img)))
            tensors.append(styled_img)
        
        return torch.cat(tensors, dim=0)


NODE_NAME = 'Image Convert [RvTools]'
NODE_DESC = 'Image Convert'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_ImageConvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}