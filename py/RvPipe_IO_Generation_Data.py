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

from typing import Optional, Any
from ..core import CATEGORY
from ..core import AnyType

any = AnyType("*")

# original code is taken from rgthree context utils
_all_context_input_output_data = {
    "pipe": ("pipe", "pipe", "pipe"),
    "steps": ("steps", "INT", "steps"),
    "cfg": ("cfg", "FLOAT", "cfg"),
    "sampler_name": ("sampler_name", "STRING", "sampler_name"),
    "scheduler": ("scheduler", "STRING", "scheduler"),
    "denoise": ("denoise", "FLOAT", "denoise"),
    "clip_skip": ("clip_skip", "INT", "clip_skip"),    
    "seed": ("seed", "INT", "seed"),
    "width": ("width", "INT", "width"),
    "height": ("height", "INT", "height"),
    "text_pos": ("text_pos", "STRING", "text_pos"),
    "text_neg": ("text_neg", "STRING", "text_neg"),
    "model_name": ("model_name", "STRING", "model_name"),
    "vae_name": ("vae_name", "STRING", "vae_name"),
    "lora_names": ("lora_names", "STRING", "lora_names"),

}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler_name", "scheduler"]

def _create_context_data(input_list=None):
    if input_list is None:
        input_list = _all_context_input_output_data.keys()
    list_ctx_return_types = []
    list_ctx_return_names = []
    ctx_optional_inputs = {}
    for inp in input_list:
        data = _all_context_input_output_data[inp]
        list_ctx_return_types.append(data[1])
        list_ctx_return_names.append(data[2])
        # Add tooltips for UI clarity
        tooltip = f"Optional input for '{data[0]}'. Accepts type: {data[1]}."
        ctx_optional_inputs[data[0]] = tuple(
            [data[1], {"forceInput": True, "tooltip": tooltip}] if data[1] in force_input_types or data[0] in force_input_names else [data[1], {"tooltip": tooltip}]
        )
    ctx_return_types = tuple(list_ctx_return_types)
    ctx_return_names = tuple(list_ctx_return_names)
    return (ctx_optional_inputs, ctx_return_types, ctx_return_names)

ALL_CTX_OPTIONAL_INPUTS, ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES = _create_context_data()

_original_ctx_inputs_list = [
    "pipe"
]
ORIG_CTX_OPTIONAL_INPUTS, ORIG_CTX_RETURN_TYPES, ORIG_CTX_RETURN_NAMES = _create_context_data(_original_ctx_inputs_list)

def new_context(pipe: Optional[dict[Any, Any]] = None, **kwargs) -> dict:
    context = pipe if pipe is not None else None
    new_ctx = {}
    for key in _all_context_input_output_data:
        if key == "pipe":
            continue
        v = kwargs.get(key, None)
        if v is not None:
            new_ctx[key] = v
        elif context is not None and key in context:
            new_ctx[key] = context[key]
        else:
            new_ctx[key] = None
    return new_ctx

def get_context_return_tuple(ctx: dict, inputs_list=None) -> tuple:
    if inputs_list is None:
        inputs_list = _all_context_input_output_data.keys()
    tup_list: list[Any] = [ctx]
    for key in inputs_list:
        if key == "pipe":
            continue
        tup_list.append(ctx[key] if ctx is not None and key in ctx else None)
    return tuple(tup_list)

class RvPipe_Generation_Data:
    # Node class for passing through generation data and a pipe context.
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": ALL_CTX_OPTIONAL_INPUTS,
            "hidden": {},
        }

    RETURN_TYPES = ALL_CTX_RETURN_TYPES
    RETURN_NAMES = ALL_CTX_RETURN_NAMES
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None, **kwargs) -> tuple:
        # Passes through the pipe context and up to 12 'Any' type channels.
        # Returns a tuple of all outputs in the correct order.
        ctx = new_context(pipe, **kwargs)
        return get_context_return_tuple(ctx)


NODE_NAME = "Generation Data [RvTools]"
NODE_DESC = "Generation Data"

NODE_CLASS_MAPPINGS = {NODE_NAME: RvPipe_Generation_Data}

NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: NODE_DESC}
