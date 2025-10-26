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
#

import json
import os
import random
import numpy as np
import folder_paths
from PIL import Image
from typing import List, Dict, Any

from ..core import CATEGORY
from ..core import AnyType

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

any = AnyType("*")

class RvTools_ShowAny:
    """
    Display any type of data as formatted text output.
    Accepts any input type and converts it to readable text format.
    Automatically detects and previews IMAGE tensors.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_showany_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "show_images": (["hide", "show"],{"default": "hide", "tooltip": "Show or hide image previews for IMAGE tensors"}),
            },
            "optional": {
                "anything": (any, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "show_value"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def _is_mask_tensor(self, tensor):
        """Check if a tensor is a MASK tensor (2D or 3D)"""
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            return False
        # Mask format: [height, width] or [batch, height, width]
        return tensor.dim() in [2, 3] and tensor.shape[-1] > 1 and tensor.shape[-2] > 1

    def _is_image_tensor(self, tensor):
        """Check if a tensor is an IMAGE tensor (4D with valid channels)"""
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            return False
        # Image format: [batch, height, width, channels]
        return tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]
    
    def _save_image_preview(self, image_tensor):
        """Save image tensor to temp folder and return metadata for preview"""
        results = []
        
        filename_prefix = "ShowAny" + self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, image_tensor.shape[2], image_tensor.shape[1])
        
        # Save each image in the batch
        for batch_number, image in enumerate(image_tensor):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        
        return results

    def _save_mask_preview(self, mask_tensor):
        """Save mask tensor to temp folder and return metadata for preview"""
        results: List[Dict[str, str]] = []
        
        filename_prefix = "ShowAnyMask" + self.prefix_append
        
        # Convert to numpy and get dimensions
        mask_np = mask_tensor.cpu().numpy()
        
        if mask_np.ndim == 2:
            height, width = mask_np.shape
            masks = [mask_np]
        elif mask_np.ndim == 3:
            height, width = mask_np.shape[-2], mask_np.shape[-1]
            masks = [mask_np[i] for i in range(mask_np.shape[0])]
        else:
            return results  # invalid shape
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, width, height)
        
        # Save each mask in the batch as grayscale image
        for batch_number, mask in enumerate(masks):
            mask_normalized = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(mask_normalized, mode='L')
            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        
        return results


    def _format_tensor(self, tensor):
        """Format tensor for display - show actual values for small tensors, summary for large ones"""
        shape = list(tensor.shape)
        dtype = tensor.dtype
        device = tensor.device
        
        # Calculate total elements
        total_elements = tensor.numel()
        
        # For very small tensors (<=20 elements), show full data
        if total_elements <= 20:
            tensor_str = str(tensor)
            return f"Tensor(shape={shape}, dtype={dtype}, device={device})\n{tensor_str}"
        
        # For small tensors (<=100 elements), show summary
        elif total_elements <= 100:
            # Convert to numpy for better formatting
            np_array = tensor.cpu().numpy()
            with np.printoptions(precision=4, suppress=True, threshold=100):
                tensor_str = str(np_array)
            return f"Tensor(shape={shape}, dtype={dtype}, device={device})\n{tensor_str}"
        
        # For larger tensors, show shape, stats, and sample
        else:
            # Get statistics
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.float().mean().item()
            
            # Get a small sample from the tensor (first few elements)
            if tensor.ndim == 1:
                sample = tensor[:5]
            elif tensor.ndim == 2:
                sample = tensor[:3, :3]
            elif tensor.ndim == 3:
                sample = tensor[:2, :2, :2]
            else:  # 4D or higher
                sample = tensor[:1, :2, :2, :2]
            
            sample_str = str(sample.cpu().numpy())
            
            return (f"Tensor(shape={shape}, dtype={dtype}, device={device})\n"
                   f"Stats: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}\n"
                   f"Sample:\n{sample_str}\n...")

    def show_value(self, show_images, unique_id=None, extra_pnginfo=None, **kwargs):
        """
        Convert any input to displayable text format.
        Handles strings, numbers, lists, dicts, tensors, and other objects.
        Previews IMAGE tensors when show_images is enabled.
        """
        # Extract show_images parameter (it's a list)
        show_images_enabled = show_images[0] == "show" if isinstance(show_images, list) else show_images == "show"
        
        original_values = []  # Keep original values for pass-through
        display_values = []   # Create display strings for UI
        image_results = []    # Store image preview metadata
        
        if "anything" in kwargs:
            for val in kwargs['anything']:
                # Always store the original value for output
                original_values.append(val)
                
                try:
                    # Create display string based on type
                    if isinstance(val, str):
                        display_values.append(val)
                    elif isinstance(val, (int, float, bool)):
                        display_values.append(str(val))
                    elif isinstance(val, list):
                        # For lists, display as-is
                        if len(val) > 0 and TORCH_AVAILABLE and all(isinstance(item, torch.Tensor) for item in val):
                            # Check if it's a list of image tensors
                            if all(self._is_image_tensor(item) for item in val):
                                # List of images - show raw list representation
                                display_values.append(str(val))
                                # Save and preview each image if enabled
                                if show_images_enabled:
                                    for img in val:
                                        image_previews = self._save_image_preview(img)
                                        image_results.extend(image_previews)

                            elif all(self._is_mask_tensor(item) for item in val):
                                # List of masks - show raw list representation
                                display_values.append(str(val))
                                # Save and preview each mask if enabled
                                if show_images_enabled:
                                    for mask in val:
                                        mask_previews = self._save_mask_preview(mask)
                                        image_results.extend(mask_previews)            
                            else:
                                # List of other tensors - show raw list
                                display_values.append(str(val))
                        elif all(isinstance(item, (str, int, float, bool)) for item in val):
                            # List of primitives - show as formatted string
                            display_values.append(str(val))
                        else:
                            display_values.append(str(val))
                    # Handle torch tensors
                    elif TORCH_AVAILABLE and isinstance(val, torch.Tensor):
                        # Check if it's an IMAGE tensor
                        if self._is_image_tensor(val):
                            # Show raw tensor representation
                            display_values.append(str(val))
                            # Save and preview if enabled (convert for image display only)
                            if show_images_enabled:
                                image_previews = self._save_image_preview(val)
                                image_results.extend(image_previews)
                        elif all(self._is_mask_tensor(item) for item in val):
                                # List of masks - show raw list representation
                                display_values.append(str(val))
                                # Save and preview each mask if enabled
                                if show_images_enabled:
                                    for mask in val:
                                        mask_previews = self._save_mask_preview(mask)
                                        image_results.extend(mask_previews)        
                        else:
                            # Regular tensor - show raw representation
                            display_values.append(str(val))
                    # Handle tuples (conditioning is often a tuple)
                    elif isinstance(val, tuple):
                        if len(val) > 0 and TORCH_AVAILABLE and isinstance(val[0], torch.Tensor):
                            # This is likely conditioning or similar
                            tensor_shapes = [list(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in val]
                            tuple_info = f"Tuple[{len(val)} items: {tensor_shapes}]"
                            display_values.append(tuple_info)
                        else:
                            display_values.append(str(val))
                    # Handle dicts
                    elif isinstance(val, dict):
                        # Show the actual dict representation (unchanged)
                        display_values.append(str(val))
                        
                        # Check if dict contains image or mask tensors and preview them
                        if show_images_enabled and TORCH_AVAILABLE:
                            for dict_key, dict_value in val.items():
                                # Check for image tensors in dict values
                                if isinstance(dict_value, torch.Tensor) and self._is_image_tensor(dict_value):
                                    image_previews = self._save_image_preview(dict_value)
                                    image_results.extend(image_previews)
                                # Check for mask tensors in dict values
                                elif isinstance(dict_value, torch.Tensor) and self._is_mask_tensor(dict_value):
                                    mask_previews = self._save_mask_preview(dict_value)
                                    image_results.extend(mask_previews)
                    else:
                        # Try to serialize to JSON
                        try:
                            json_val = json.dumps(val)
                            display_values.append(json_val)
                        except (TypeError, ValueError):
                            # If JSON serialization fails, use string representation
                            display_values.append(str(val))
                except Exception as e:
                    # Fallback to type name for display
                    display_values.append(f"<{type(val).__name__}>")

        # Ensure all display values are strings for UI
        string_values = []
        for v in display_values:
            if isinstance(v, str):
                string_values.append(v)
            else:
                string_values.append(str(v))
        
        # Update workflow metadata with the display values and show_images setting
        if extra_pnginfo:
            if isinstance(extra_pnginfo[0], dict) and "workflow" in extra_pnginfo[0]:
                workflow = extra_pnginfo[0]["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
                if node:
                    # Save both the text values and the show_images setting
                    node["widgets_values"] = [show_images[0] if isinstance(show_images, list) else show_images]

        # Build UI response - only include images if show_images is enabled
        ui_response = {"text": string_values}
        if show_images_enabled and image_results:
            ui_response["images"] = image_results

        # Return original values for pass-through, display strings and images for UI
        if isinstance(original_values, list) and len(original_values) == 1:
            return {
                "ui": ui_response,
                "result": (original_values[0],),  # Pass through ORIGINAL value
            }
        else:
            return {
                "ui": ui_response,
                "result": (original_values,),  # Pass through ORIGINAL values
            }


NODE_NAME = 'Show Any [RvTools]'
NODE_DESC = 'Show Any'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvTools_ShowAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
