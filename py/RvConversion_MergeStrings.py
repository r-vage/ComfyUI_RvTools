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

import re
from ..core import CATEGORY
from typing import Any, Dict, Tuple

class RvConversion_MergeStrings:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1, "tooltip": "Number of string inputs to merge. Only the first 'inputcount' string_X inputs will be used."}),
                "Delimiter": ("STRING", {"default": ", ", "tooltip": "Delimiter to use between strings when merging. Use \\n for newline."}),
                "return_as_list": ("BOOLEAN", {"default": False, "tooltip": "If true, return list of individual strings; if false, return single merged string."}),
            },
            "optional": {
                "string_1": ("STRING", {"forceInput": True, "default": "", "tooltip": "String input #1."}),
                "string_2": ("STRING", {"forceInput": True, "default": "", "tooltip": "String input #2."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self, inputcount: int, Delimiter: str = ", ", return_as_list: bool = False, **kwargs) -> Tuple[Any]:
        text_inputs = []

        # Handle special case for literal newlines
        if Delimiter in ("\n", "\\n"):
            Delimiter = "\n"

        # Collect and process strings from the first 'inputcount' inputs
        for i in range(1, min(inputcount, 64) + 1):
            key = f"string_{i}"
            v = kwargs.get(key, "")
            if isinstance(v, str):
                v = v.strip()  # Trim whitespace
                v = v.rstrip('.,;:!?')  # Remove trailing punctuation
                if v:  # Only add if not empty after processing
                    text_inputs.append(v)

        if return_as_list:
            # Return list of strings
            return (text_inputs,)
        else:
            # Merge strings
            merged_text = Delimiter.join(text_inputs)
            # Replace line breaks with spaces for prompt compatibility
            merged_text = re.sub(r"[\r\n]+", " ", merged_text)
            return ([merged_text],)

NODE_NAME = 'Merge Strings [Eclipse]'
NODE_DESC = 'Merge Strings'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_MergeStrings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}