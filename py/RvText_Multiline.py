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

from ..core import CATEGORY

class RvText_Multiline:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": ""}, "Multiline string input.\nSplits into a list of lines and returns the full string joined by commas.\nUseful for prompt construction, text processing, or passing lists to downstream nodes."),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("string", "string_list",)

    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "execute"

    def execute(self, string=None):
        # Outputs the input multiline string as a single joined string and as a list of lines.
        # Handles None, empty, and whitespace-only input robustly.
        if not isinstance(string, str) or not string or string.isspace():
            return ("", [""])

        # Strip and split the input
        string = string.strip()
        string_list = string.split('\n')

        # Filter out empty lines and strip whitespace
        string_list = [line.strip() for line in string_list if line.strip()]

        # If no valid lines found, return empty
        if not string_list:
            return ("", [""])

        # Output: fallback for single item
        if len(string_list) == 1:
            return (string_list[0], string_list)
        joined_string = " ".join(string_list)
        return (joined_string, string_list)

NODE_NAME = 'String Multiline [RvTools]'
NODE_DESC = 'String Multiline'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvText_Multiline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}