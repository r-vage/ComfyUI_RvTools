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
from ..core import CATEGORY, AnyType

any = AnyType("*")

class RvConversion_WidgetToString:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value
    RETURN_TYPES = ("STRING", )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Node ID to extract widget value from."}),
                "widget_name": ("STRING", {"multiline": False, "tooltip": "Name of the widget to extract."}),
                "return_all": ("BOOLEAN", {"default": False, "tooltip": "Return all widget values as a formatted string."}),
            },
            "optional": {
                "any_input": (any, {}),
                "node_title": ("STRING", {"multiline": False, "tooltip": "Node title to match instead of ID."}),
                "allowed_float_decimals": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Number of decimal places to display for float values."}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def execute(
        self,
        id: int,
        widget_name: str,
        extra_pnginfo,
        prompt,
        unique_id,
        return_all: bool = False,
        any_input=None,
        node_title: str = "",
        allowed_float_decimals: int = 2
    ) -> tuple[str]:
        workflow = extra_pnginfo["workflow"]
        results = []
        node_id = None
        link_id = None
        link_to_node_map = {}

        for node in workflow["nodes"]:
            if node_title:
                if "title" in node and node["title"] == node_title:
                    node_id = node["id"]
                    break
            elif id != 0:
                if node["id"] == id:
                    node_id = id
                    break
            elif any_input is not None:
                if node["type"] == "Widget to String [RvTools]" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        if node_input["name"] == "any_input":
                            link_id = node_input["link"]
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("No matching node found for the given title or id")

        values = prompt[str(node_id)]
        if "inputs" in values:
            if return_all:
                formatted_items = []
                for k, v in values["inputs"].items():
                    if isinstance(v, float):
                        item = f"{k}: {v:.{allowed_float_decimals}f}"
                    else:
                        item = f"{k}: {str(v)}"
                    formatted_items.append(item)
                result = ', '.join(formatted_items)
                # Replace all line breaks with spaces for prompt output
                result = re.sub(r"[\r\n]+", " ", result)
                results.append(result)
            elif widget_name in values["inputs"]:
                v = values["inputs"][widget_name]
                if isinstance(v, float):
                    v = f"{v:.{allowed_float_decimals}f}"
                else:
                    v = str(v)
                # Replace all line breaks with spaces for prompt output
                v = re.sub(r"[\r\n]+", " ", v)
                return (v, )
            else:
                raise NameError(f"Widget not found: {node_id}.{widget_name}")
        return (', '.join(results).strip(', '), )

NODE_NAME = 'Widget to String [RvTools]'
NODE_DESC = 'Widget to String'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvConversion_WidgetToString
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}