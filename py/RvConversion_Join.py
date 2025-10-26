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
import re
from ..core import CATEGORY, AnyType
from typing import Any, Dict, Tuple

any_type = AnyType("*")

class RvConversion_Join:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1, "tooltip": "Number of inputs to join. Only the first 'inputcount' input_X values will be used."}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ", ", "tooltip": "Delimiter for STRING types. Use \\n for newline. Ignored for IMAGE/MASK."}),
                "input_1": (any_type, {"forceInput": True, "tooltip": "Input #1."}),
                "input_2": (any_type, {"forceInput": True, "tooltip": "Input #2."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"

    def execute(self, inputcount: int, delimiter: str = ", ", **kwargs) -> Tuple[Any]:
        inputs = []
        
        # Collect inputs
        for i in range(1, min(inputcount, 64) + 1):
            key = f"input_{i}"
            v = kwargs.get(key)
            if v is not None:
                inputs.append(v)
        
        if not inputs:
            return (None,)
        
        # Detect type from first input
        first_input = inputs[0]
        
        # Handle IMAGE type
        if isinstance(first_input, torch.Tensor) and first_input.ndim == 4:
            return self._join_images(inputs)
        
        # Handle MASK type
        elif isinstance(first_input, torch.Tensor) and first_input.ndim in (2, 3):
            return self._join_masks(inputs)
        
        # Handle STRING type
        elif isinstance(first_input, str):
            return self._join_strings(inputs, delimiter)
        
        # Handle INT type
        elif isinstance(first_input, int):
            return self._join_primitives(inputs, delimiter)
        
        # Handle FLOAT type
        elif isinstance(first_input, float):
            return self._join_primitives(inputs, delimiter)
        
        # Handle lists (convert to string representation)
        elif isinstance(first_input, (list, tuple)):
            return self._join_primitives(inputs, delimiter)
        
        # Fallback: return first input
        print(f"[RvTools Join] Unknown type: {type(first_input)}, returning first input")
        return (first_input,)
    
    def _join_images(self, inputs):
        """Join IMAGE tensors into a batch, resizing to match first image dimensions"""
        tensors = []
        for img in inputs:
            if isinstance(img, torch.Tensor) and img.ndim == 4:
                tensors.append(img)
        
        if not tensors:
            return (None,)
        
        # Get target dimensions from first image
        # Shape is [B, H, W, C]
        target_height = tensors[0].shape[1]
        target_width = tensors[0].shape[2]
        
        # Resize all images to match target dimensions
        resized_tensors = []
        for tensor in tensors:
            if tensor.shape[1] != target_height or tensor.shape[2] != target_width:
                # Need to resize - convert to [B, C, H, W] for interpolate
                # Current shape: [B, H, W, C]
                tensor_bchw = tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
                
                # Resize using bilinear interpolation
                resized = torch.nn.functional.interpolate(
                    tensor_bchw, 
                    size=(target_height, target_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Convert back to [B, H, W, C]
                tensor = resized.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            resized_tensors.append(tensor)
        
        # Concatenate along batch dimension
        result = torch.cat(resized_tensors, dim=0)
        return (result,)
    
    def _join_masks(self, inputs):
        """Join MASK tensors into a batch, resizing to match first mask dimensions"""
        tensors = []
        for mask in inputs:
            if isinstance(mask, torch.Tensor):
                # Ensure 3D format [B, H, W]
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 4:
                    mask = mask.squeeze(0)
                tensors.append(mask)
        
        if not tensors:
            return (None,)
        
        # Get target dimensions from first mask
        # Shape is [B, H, W]
        target_height = tensors[0].shape[1]
        target_width = tensors[0].shape[2]
        
        # Resize all masks to match target dimensions
        resized_tensors = []
        for tensor in tensors:
            if tensor.shape[1] != target_height or tensor.shape[2] != target_width:
                # Need to resize - add channel dimension for interpolate
                tensor_bchw = tensor.unsqueeze(1)  # [B, 1, H, W]
                
                # Resize using bilinear interpolation
                resized = torch.nn.functional.interpolate(
                    tensor_bchw, 
                    size=(target_height, target_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Remove channel dimension
                tensor = resized.squeeze(1)  # [B, H, W]
            
            resized_tensors.append(tensor)
        
        # Concatenate along batch dimension
        result = torch.cat(resized_tensors, dim=0)
        return (result,)
    
    def _join_strings(self, inputs, delimiter: str):
        """Join STRING values with delimiter"""
        # Handle special case for literal newlines
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"
        
        text_inputs = []
        for v in inputs:
            if isinstance(v, str):
                v = v.strip()  # Trim whitespace
                v = v.rstrip('.,;:!?')  # Remove trailing punctuation
                if v:  # Only add if not empty after processing
                    text_inputs.append(v)
        
        if not text_inputs:
            return ("",)
        
        # Merge strings
        merged_text = delimiter.join(text_inputs)
        # Replace line breaks with spaces for prompt compatibility
        merged_text = re.sub(r"[\r\n]+", " ", merged_text)
        return (merged_text,)
    
    def _join_primitives(self, inputs, delimiter: str):
        """Join INT/FLOAT/LIST values as comma-separated string"""
        # Handle special case for literal newlines
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"
        
        text_inputs = []
        for v in inputs:
            if v is not None:
                text_inputs.append(str(v))
        
        if not text_inputs:
            return ("",)
        
        # Merge as string
        merged_text = delimiter.join(text_inputs)
        return (merged_text,)


NODE_NAME = 'Join [RvTools]'
NODE_DESC = 'Join'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_Join
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
