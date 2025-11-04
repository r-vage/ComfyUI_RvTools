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

class RvText_DualText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "txt_pos": ("STRING", {"multiline": True, "default": ""}),
                "txt_neg": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("txt_pos", "txt_neg",)

    FUNCTION = "execute"

    def execute(self, txt_pos: str, txt_neg: str):
        txt_pos = txt_pos.strip()
        txt_neg = txt_neg.strip()
        
        return (txt_pos, txt_neg)

NODE_NAME = 'String Dual [Eclipse]'
NODE_DESC = 'String Dual'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvText_DualText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}