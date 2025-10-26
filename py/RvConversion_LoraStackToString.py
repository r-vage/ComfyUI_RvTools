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

class RvConversion_LoraStackToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {},
                    "List of LoRA tuples (STR, FLOAT1, FLOAT2).\nReturns a space-separated string in <lora:...> format."),
                "remove_weight": ("BOOLEAN", {"default": False}, "If true, removes the last 2 elements from each tuple, using only the LoRA name.")
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LoRA string",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value

    def convert(self, lora_stack, remove_weight):
        # Type safety: handle None and non-iterable input
        if lora_stack is None or not hasattr(lora_stack, "__iter__"):
            return ("",)
        try:
            if remove_weight:
                output = ' '.join(
                    f"<lora:{str(tup[0])}>"
                    for tup in lora_stack
                    if isinstance(tup, (list, tuple)) and len(tup) >= 1
                )
            else:
                output = ' '.join(
                    f"<lora:{str(tup[0])}:{str(tup[1])}:{str(tup[2])}>"
                    for tup in lora_stack
                    if isinstance(tup, (list, tuple)) and len(tup) >= 3
                )
            return (output,)
        except Exception:
            return ("",)

NODE_NAME = "Lora Stack to String [RvTools]"
NODE_DESC = "Lora Stack to String"

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_LoraStackToString
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}

