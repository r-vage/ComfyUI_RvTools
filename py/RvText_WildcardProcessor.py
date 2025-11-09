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
# Supports two processing modes:
# - populate: Expands all wildcards and options. Seed controls output - change seed for new output, fix seed for consistent output
# - fixed: Uses populated_text as-is, ignoring wildcards
#
# Special seed values (from eclipse-seed.js extension):
#   -1: Randomize each time (generates new random seed)
#   -2: Increment from last seed
#   -3: Decrement from last seed


import os
import logging
from typing import Any, Dict, Tuple, Optional, List

from ..core import CATEGORY
from ..core.wildcard_engine import wildcard_load, process


class RvText_WildcardProcessor:
    # A wildcard text processor that expands wildcard patterns and options.
    #
    # Wildcard Syntax:
    # - {option1|option2|option3} - Random selection from options
    # - __keyword__ - Reference to wildcard group
    # - 2$$opt1|opt2 - Select N items with separator
    # - 1-3$$opt1|opt2|opt3 - Select range N-M items
    # - 1.0::item1|2.0::item2 - Probability weights (2x likelihood for item2)
    # - 3#__keyword__ - Use keyword 3 times

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": ("STRING", {
                    "multiline": True,
                    "default": "Try using __wildcard__ or {option1|option2}",
                    "dynamicPrompts": False,
                    "tooltip": "Enter a prompt using wildcard syntax."
                }),
                "populated_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "The actual value processed from 'wildcard_text'. In 'populate' mode, this is auto-updated. In 'fixed' mode, you can edit this value."
                }),
                "mode": (["populate", "fixed"], {
                    "default": "populate",
                    "tooltip": "populate: Auto-processes wildcard_text based on seed. Change seed for new output, fix seed for consistent output.\nfixed: Uses populated_text as-is, you can edit it"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": -3,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed controls wildcard expansion in populate mode.\nSpecial values: -1=randomize each time, -2=increment from last, -3=decrement from last\nNormal: change to generate new outputs, fix to keep same output.\nCan be connected to seed nodes like rgthree's Seed node."
                }),
            },
            "optional": {
                # Combo will be populated dynamically by JavaScript
                "wildcards": (["Select a Wildcard"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value

    OUTPUT_NODE = False

    # NO IS_CHANGED - we preprocess wildcards server-side like Impact Pack does
    # The prompt handler in server_endpoints.py processes wildcards before execution

    def execute(
        self,
        wildcard_text: str,
        populated_text: str,
        mode: str,
        seed: int,
        wildcards: str = "Select a Wildcard"
    ) -> Dict[str, Any]:

        try:
            # The server-side prompt handler (onprompt_populate_wildcards) already processed
            # wildcards and updated populated_text before execution in populate mode.
            # So we just use populated_text directly.
            
            # In "populate" mode: populated_text was already processed by server handler
            # In "fixed" mode: populated_text contains manually edited text
            result = populated_text
            
            # Add selected wildcard if not "Select a Wildcard"
            if wildcards and wildcards != "Select a Wildcard":
                if result and not result.endswith('\n'):
                    result += '\n'
                result += wildcards
                # Process the added wildcard
                result = process(result, seed=seed)
            
            # Return both the result and UI data for onExecuted handler
            return {
                "ui": {
                    "text": [result],
                    "seed": [seed]
                },
                "result": (result,)
            }

        except Exception as e:
            logging.error(f"[Eclipse Wildcard] Error in execute: {e}")
            return {
                "ui": {"text": [populated_text]},
                "result": (populated_text,)
            }

    @staticmethod
    def load_wildcard_path(path: Optional[str] = None) -> None:
        if path is None:
            # Default to root/wildcards/ in workspace
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "wildcards"
            )

        wildcard_load(path)
        logging.info(f"[Eclipse Wildcard] Loaded wildcards from: {path}")


# Ensure wildcard engine is initialized on import
_wildcard_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "wildcards"
)
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
