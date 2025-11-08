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
# RvText_WildcardProcessor - Processes text with wildcard syntax
#
# Server-side wildcard processing (simplified from Impact Pack):
# - populate mode: Server processes wildcard_text before execution, updates populated_text, then switches to fixed
# - fixed mode: Uses populated_text as-is, ignoring wildcard_text (stays fixed until user changes it)


import os
import logging
from typing import Any, Dict, Tuple, Optional, List

from ..core import CATEGORY
from ..core.wildcard_engine import wildcard_load, process, get_wildcard_list


class RvText_WildcardProcessor:
    """
    A wildcard text processor that expands wildcard patterns and options.
    
    Wildcard Syntax:
    - {option1|option2|option3} - Random selection from options
    - __keyword__ - Reference to wildcard group
    - 2$$opt1|opt2 - Select N items with separator
    - 1-3$$opt1|opt2|opt3 - Select range N-M items
    - 1.0::item1|2.0::item2 - Probability weights (2x likelihood for item2)
    - 3#__keyword__ - Use keyword 3 times
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Enter a prompt using wildcard syntax."
                }),
                "populated_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "The actual value passed during execution. "
                               "In 'populate' mode, this is auto-generated before execution. "
                               "In 'fixed' mode, you can edit this directly."
                }),
                "mode": (["populate", "fixed"], {
                    "default": "populate",
                    "tooltip": "populate: Before running, overwrites populated_text with processed wildcard_text. This widget cannot be edited.\n"
                               "fixed: Ignores wildcard_text and keeps populated_text as-is. You can edit populated_text in this mode. When you generate with 'populate', the mode automatically switches to 'fixed' to preserve the result."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Determines the random seed used for wildcard processing."
                }),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value

    OUTPUT_NODE = False

    @staticmethod
    def process_wildcards(**kwargs):
        """Static method for processing wildcards - used by server handler too."""
        return process(**kwargs)

    def doit(self, wildcard_text: str, populated_text: str, mode: str, seed: int, **kwargs) -> Tuple[str]:
        """
        Execute the wildcard processor.
        
        The server-side prompt handler (onprompt_populate_wildcards) already processed
        wildcards and updated populated_text before execution, so we just process
        populated_text one more time with the current seed to ensure consistency.
        """
        # Process the populated_text (which was already set by server handler)
        # This ensures wildcards in populated_text are also expanded
        processed_text = process(text=populated_text, seed=seed)
        return (processed_text,)

    @staticmethod
    def load_wildcard_path(path: Optional[str] = None) -> None:
        """Load wildcards from the specified path."""
        if path is None:
            # Use same path resolution as server_endpoints.py
            extension_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            extension_wildcard_path = os.path.join(extension_root, "wildcards")
            
            # Try to find ComfyUI root
            comfyui_root = os.path.abspath(os.path.join(extension_root, "..", ".."))
            models_wildcard_path = os.path.join(comfyui_root, "models", "wildcards")
            
            # Priority: models/wildcards, then extension/wildcards
            if os.path.exists(os.path.join(comfyui_root, "models")) and os.path.exists(models_wildcard_path):
                path = models_wildcard_path
            else:
                path = extension_wildcard_path

        wildcard_load(path)
        logging.info(f"[Eclipse Wildcard] Loaded wildcards from: {path}")


# Ensure wildcard engine is initialized on import - use same logic
def _get_initial_wildcard_path():
    extension_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    extension_wildcard_path = os.path.join(extension_root, "wildcards")
    
    comfyui_root = os.path.abspath(os.path.join(extension_root, "..", ".."))
    models_wildcard_path = os.path.join(comfyui_root, "models", "wildcards")
    
    if os.path.exists(os.path.join(comfyui_root, "models")) and os.path.exists(models_wildcard_path):
        return models_wildcard_path
    return extension_wildcard_path

_wildcard_path = _get_initial_wildcard_path()
if os.path.exists(_wildcard_path):
    RvText_WildcardProcessor.load_wildcard_path(_wildcard_path)


NODE_NAME = 'Wildcard Processor [Eclipse]'
NODE_DESC = 'Wildcard Processor'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvText_WildcardProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
