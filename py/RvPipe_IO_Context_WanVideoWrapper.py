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
from ..core import AnyType

any = AnyType("*")

# original code is taken from rgthree context utils
_all_context_input_output_data = {
    "pipe": ("pipe", "pipe", "context"),

    "torch_compile_args": ("torch_compile_args", "WANCOMPILEARGS", "torch_compile_args"),
    "block_swap_args": ("block_swap_args", "BLOCKSWAPARGS", "block_swap_args"),
    "vram_management_args": ("vram_management_args", "VRAM_MANAGEMENTARGS", "vram_management_args"),     
    "loras": ("loras", "WANVIDLORA", "loras"),

    "model": ("model", "WANVIDEOMODEL", "model"),
    "clip": ("clip", "WANTEXTENCODER", "clip"),
    "vae": ("vae", "WANVAE", "vae"),

    "text_embeds": ("text_embeds", "WANVIDEOTEXTEMBEDS", "text_embeds"),
    "image_embeds": ("image_embeds", "WANVIDIMAGE_EMBEDS", "image_embeds"),
        
    "images_input": ("images_input", "IMAGE", "images_input"),
    "images_ref_start": ("images_ref_start", "IMAGE", "images_ref_start"),
    "images_ref_new": ("images_ref_new", "IMAGE", "images_ref_new"),
    "images_ref_end": ("images_ref_end", "IMAGE", "images_ref_end"),
    "images_1st": ("images_1st", "IMAGE", "images_1st"),
    "images_loop": ("images_loop", "IMAGE", "images_loop"),
    
    "images_pp1": ("images_pp1", "IMAGE", "images_pp1"),
    "images_pp2": ("images_pp2", "IMAGE", "images_pp2"),
    "images_pp3": ("images_pp3", "IMAGE", "images_pp3"),

    "images_output": ("images_output", "IMAGE", "images_output"),

    "mask_1": ("mask_1", "MASK", "mask_1"),
    "mask_2": ("mask_2", "MASK", "mask_2"),
    "mask_3": ("mask_3", "MASK", "mask_3"),

    "steps": ("steps", "INT", "steps"),
    "cfg": ("cfg", "FLOAT", "cfg"),
    "sampler_name": ("sampler_name", any, "sampler_name"),
    "scheduler": ("scheduler", any, "scheduler"),
    "denoise": ("denoise", "FLOAT", "denoise"),
    "seed": ("seed", "INT", "seed"),

    "width": ("width", "INT", "width"),
    "height": ("height", "INT", "height"),

    "text_pos": ("text_pos", "STRING", "text_pos"),
    "text_i2p": ("text_i2p", "STRING", "text_i2p"),
    "text_neg": ("text_neg", "STRING", "text_neg"),

    "frame_rate": ("frame_rate", "FLOAT", "frame_rate"),
    "frame_load_cap": ("frame_load_cap", "INT", "frame_load_cap"),
    "context_length": ("context_length", "INT", "context_length"),
    "overlap": ("overlap", "INT", "overlap"),
    "skip_first_frames": ("skip_first_frames", "INT", "skip_first_frames"),
    "select_every_nth": ("select_every_nth", "INT", "select_every_nth"),
    "loop_idx": ("loop_idx", "INT", "loop_idx"),

    "audio": ("audio", "AUDIO", "audio"),

    "any_1": ("any_1", any, "any_1"),
    "any_2": ("any_2", any, "any_2"),

    "path": ("path", "STRING", "path"),
}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler_name", "scheduler"]

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
        tooltip = f"Optional input for '{data[0]}'. Accepts type: {data[1]}."
        ctx_optional_inputs[data[0]] = tuple(
            [data[1], {"forceInput": True, "tooltip": tooltip}] if data[1] in force_input_types or data[0] in force_input_names else [data[1], {"tooltip": tooltip}]
        )
    ctx_return_types = tuple(list_ctx_return_types)
    ctx_return_names = tuple(list_ctx_return_names)
    return (ctx_optional_inputs, ctx_optional_inputs, ctx_return_types, ctx_return_names)  # inputs and outputs

ALL_CTX_OPTIONAL_INPUTS, ALL_CTX_OPTIONAL_OUTPUTS, ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES = _create_context_data()

def new_context(pipe=None, **kwargs):
    # Creates a new context from the provided data, with an optional base ctx to start.
    # pipe can be dict or tuple
    if isinstance(pipe, tuple):
        # Assume it's the tuple from get_context_return_tuple, first is dict
        context = pipe[0] if pipe else {}
    elif isinstance(pipe, dict):
        context = pipe
    else:
        context = {}
    # Only copy known keys from pipe input, ignore unknown keys
    new_ctx = {}
    for key in _all_context_input_output_data:
        if key == "pipe":
            continue
        if key in context:
            new_ctx[key] = context[key]
    # Apply kwargs overrides for known keys
    for key in _all_context_input_output_data:
        if key == "pipe":
            continue
        v = kwargs.get(key, None)
        if v is not None:
            new_ctx[key] = v
    return new_ctx

def get_context_return_tuple(ctx, inputs_list=None):
    # Returns a tuple for returning in the order of the inputs list.
    if inputs_list is None:
        inputs_list = _all_context_input_output_data.keys()
    tup_list = [ctx]
    for key in inputs_list:
        if key == "pipe":
            continue
        tup_list.append(ctx.get(key, None))
    return tuple(tup_list)

class RvPipe_IO_Context_WanVideoWrapper:
    # Node class for passing through a context for general workflows.
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

    def execute(self, pipe=None, **kwargs):
        # Read the pipe input (dict or tuple), update with connected inputs
        ctx = new_context(pipe, **kwargs)
        # Return the updated pipe and all individual values
        return get_context_return_tuple(ctx)

NODE_NAME = 'Context Video (WVW) [RvTools]'
NODE_DESC = 'Context Video (WVW)'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_IO_Context_WanVideoWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}