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

class RvText_ReplaceString:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "String": ("STRING", {"default": "", "tooltip": "Input string to process."}),
                "Regex": ("STRING", {"default": "", "tooltip": "Regular expression pattern to match."}),
                "ReplaceWith": ("STRING", {"default": "", "tooltip": "Replacement string for matches."}),
            }
        }

    def execute(self, String: str, Regex: str, ReplaceWith: str) -> tuple[str]:
        # Replace substrings in String using Regex, then remove line breaks for prompt output.
        replaced = re.sub(Regex, ReplaceWith, String)
        replaced = re.sub(r"[\r\n]+", " ", replaced)
        return (replaced,)

NODE_NAME = 'Replace String [Eclipse]'
NODE_DESC = 'Replace String'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvText_ReplaceString
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}