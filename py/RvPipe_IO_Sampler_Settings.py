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
    "sampler_name": ("sampler_name", any, "sampler_name"),
    "scheduler": ("scheduler", any, "scheduler"),
    "guidance": ("guidance", "FLOAT", "guidance"),
    "denoise": ("denoise", "FLOAT", "denoise"),
    "sigmas_denoise": ("sigmas_denoise", "FLOAT", "sigmas_denoise"),
    "noise_strength": ("noise_strength", "FLOAT", "noise_strength"),
    "seed": ("seed", "INT", "seed"),
}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler", "scheduler"]

def _create_context_data(input_list=None):
    # Returns a tuple of context inputs, return types, and return names to use in a node's def.
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
        tooltip = f"Optional input for channel '{data[0]}'. Accepts any type."
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
    # Creates a new context from the provided data, with an optional base pipe to start.
    #
    # Priority logic:
    # 1. If pipe has _allow_overwrite=False (default): Use pipe values, kwargs as fallback
    # 2. If pipe has _allow_overwrite=True: Use kwargs (direct inputs), pipe as fallback
    context = pipe if pipe is not None else None
    new_ctx = {}
    
    # Check if pipe allows overwriting (default is False - use pipe values)
    allow_overwrite = False
    if context is not None and isinstance(context, dict):
        allow_overwrite = context.get("_allow_overwrite", False)
    
    for key in _all_context_input_output_data:
        if key == "pipe":
            continue
        
        # Get values from both sources
        kwarg_value = kwargs.get(key, None)
        pipe_value = context.get(key, None) if context is not None and key in context else None
        
        # Apply priority logic
        if allow_overwrite:
            # allow_overwrite=True: Direct inputs take priority, pipe as fallback
            if kwarg_value is not None:
                new_ctx[key] = kwarg_value
            elif pipe_value is not None:
                new_ctx[key] = pipe_value
            else:
                new_ctx[key] = None
        else:
            # allow_overwrite=False (default): Pipe values take priority, kwargs as fallback
            if pipe_value is not None:
                new_ctx[key] = pipe_value
            elif kwarg_value is not None:
                new_ctx[key] = kwarg_value
            else:
                new_ctx[key] = None
    
    return new_ctx

def get_context_return_tuple(ctx: dict, inputs_list=None) -> tuple:
    # Returns a tuple for returning in the order of the inputs list.
    if inputs_list is None:
        inputs_list = _all_context_input_output_data.keys()
    tup_list: list[Any] = [ctx]
    for key in inputs_list:
        if key == "pipe":
            continue
        tup_list.append(ctx[key] if ctx is not None and key in ctx else None)
    return tuple(tup_list)

class RvPipe_IO_Sampler_Settings:
    # Node class for passing through up to 12 'Any' type channels and a pipe context.
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

NODE_NAME = 'Pipe IO Sampler Settings [Eclipse]'
NODE_DESC = 'Pipe IO Sampler Settings'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_IO_Sampler_Settings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}