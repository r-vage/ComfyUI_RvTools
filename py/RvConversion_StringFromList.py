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

from ..core import CATEGORY, cstr

def wrapIndex(index, length):
    # Calculate wrapped index and number of wraps
    if length <= 0:
        cstr("Invalid list length, returning 0.").error.print()
        return 0, 0
        
    # Convert to integer and handle wrap-around
    index = int(index)
    index_mod = ((index % length) + length) % length  # Handles negative indices correctly
    wraps = index // length if length > 0 else 0
    return index_mod, wraps

class RvConversion_StringFromList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("STRING", {"forceInput": True, "tooltip": "List of strings to select from."}),
                "index": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1, "tooltip": "Index to select (supports wrap-around)."}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("list item", "size", "wraps")

    # Ensure node receives list inputs as a single list (not mapped per-item)
    INPUT_IS_LIST = True
    # Output is a single STRING, not a list
    OUTPUT_IS_LIST = (True, False, True)

    FUNCTION = "execute"

    def execute(self, list_input, index):
        # Selects a string from a list by index, with wrap-around and reporting of list size and wraps.
        # Handles None and empty input robustly.
        #
        # Type safety: ensure valid list and index
        if not isinstance(list_input, (list, str)) or not list_input:
            return ([], 0, [])
        if isinstance(list_input, str):
            list_input = [list_input]
        length = len(list_input)
        wraps_list = []
        item_list = []
        indices = index if isinstance(index, list) else [index]
        for i in indices:
            if not isinstance(i, int):
                i = 0
            index_mod, wraps = wrapIndex(i, length)
            if 0 <= index_mod < length:
                wraps_list.append(wraps)
                item_list.append(list_input[index_mod])
            else:
                wraps_list.append(0)
                item_list.append("")
        return (item_list, length, wraps_list)

NODE_NAME = 'String from List [RvTools]'
NODE_DESC = 'String from List'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_StringFromList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}