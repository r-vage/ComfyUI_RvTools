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

from ..core import CATEGORY, AnyType
from typing import Any

any_type = AnyType("*")


class RvConversion_ConvertPrimitive:
    #Convert any input to primitive types: STRING, INT, FLOAT, or COMBO.
    #Handles single values only - does not accept list inputs.
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type, {"forceInput": True, "tooltip": "Any value to convert (single values only)"}),
                "convert_to": (["STRING", "INT", "FLOAT", "COMBO"], 
                              {"default": "STRING", "tooltip": "Target primitive type"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"

    def execute(self, input: Any, convert_to: str) -> tuple[Any, ...]:
        #Convert input to the specified primitive type.
        #Returns the converted value based on convert_to selection.
        #Handle COMBO type separately (needs special return format)
        if convert_to == "COMBO":
            return self._convert_to_combo(input)
        
        # Check for list/tuple input and reject it
        if isinstance(input, (list, tuple)):
            print(f"[RvTools ConvertPrimitive] Warning: List/tuple input detected. Use ConvertToList node first to extract values.")
            # Take first element as fallback
            if len(input) > 0:
                input = input[0]
            else:
                input = ""
        
        try:
            result: Any
            if convert_to == "STRING":
                # Convert to string and clean it
                if isinstance(input, dict):
                    result = str(input)
                elif isinstance(input, bool):
                    result = "true" if input else "false"
                else:
                    result = str(input)
                
                # Remove newlines, tabs, and extra whitespace
                result = result.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                # Collapse multiple spaces into single space
                result = ' '.join(result.split())
                
                return (result,)
            
            elif convert_to == "INT":
                # Convert to int
                if isinstance(input, bool):
                    result = 1 if input else 0
                elif isinstance(input, (int, float)):
                    result = int(input)
                elif isinstance(input, str):
                    # Try to parse string to int
                    cleaned = input.strip().lower()
                    if cleaned in ("true", "yes", "on", "1"):
                        result = 1
                    elif cleaned in ("false", "no", "off", "0"):
                        result = 0
                    else:
                        # Try parsing as number
                        result = int(float(cleaned))
                else:
                    result = 0
                return (result,)
            
            elif convert_to == "FLOAT":
                # Convert to float
                if isinstance(input, bool):
                    result = 1.0 if input else 0.0
                elif isinstance(input, (int, float)):
                    result = float(input)
                elif isinstance(input, str):
                    # Try to parse string to float
                    cleaned = input.strip().lower()
                    if cleaned in ("true", "yes", "on"):
                        result = 1.0
                    elif cleaned in ("false", "no", "off"):
                        result = 0.0
                    else:
                        result = float(cleaned)
                else:
                    result = 0.0
                return (result,)
            
        except (ValueError, TypeError) as e:
            # If conversion fails, return defaults
            print(f"[RvTools ConvertPrimitive] Conversion error: {e}")
            if convert_to == "STRING":
                return ("",)
            elif convert_to == "INT":
                return (0,)
            elif convert_to == "FLOAT":
                return (0.0,)
        
        return ("",)
    
    def _scalar_to_str(self, value):
        #Convert a scalar value to a safe string for combo use.
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        
        # Convert to string and check if it's an unhelpful object representation
        try:
            result = f"{value}"
            # If we get an object address like "<module.Class object at 0x...>",
            # note that the actual value cannot be extracted
            if '<' in result and 'object at 0x' in result:
                # At least make it clear this is an object reference, not a value
                return f"[Object: {result}]"
            return result
        except Exception:
            return repr(value)
    
    def _convert_to_combo(self, input):
        #Convert input to COMBO format: (selected_value, [options_list])
        from collections.abc import Iterable
        
        # For iterable inputs (not string/bytes), create options list from elements
        if isinstance(input, Iterable) and not isinstance(input, (str, bytes, bytearray)):
            try:
                # Convert each element to string
                options = [self._scalar_to_str(item) for item in input]
                if len(options) == 0:
                    # Empty list -> return empty string combo
                    return (("", [""]),)
                # First element is selected by default
                return ((options[0], options),)
            except Exception as e:
                print(f"[RvTools ConvertPrimitive] COMBO conversion from iterable failed: {e}")
                return (("", [""]),)
        else:
            # Single value -> create combo with single option
            str_val = self._scalar_to_str(input)
            return ((str_val, [str_val]),)


NODE_NAME = "Convert Primitive [RvTools]"
NODE_DESC = "Convert Primitive"

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvConversion_ConvertPrimitive
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
