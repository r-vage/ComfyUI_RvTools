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

# RvPipe_Out_LM_AdvancedOptions - Advanced parameter pipe for Language Model nodes
#
# Provides advanced generation parameters through a pipe output that can be
# connected to the optional pipe_opt input of the Smart Language Model Loader.
# This allows adjusting advanced parameters without maintaining both basic and advanced nodes.

import os
import json
from ..core import CATEGORY

# Default parameter values (fallback if config file missing)
DEFAULT_PARAMS = {
    "QwenVL": {
        "device": "cuda",
        "use_torch_compile": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "num_beams": 3,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "frame_count": 8
    },
    "Florence2": {
        "device": "cuda",
        "use_torch_compile": False,
        "num_beams": 3,
        "do_sample": True
    },
    "LLM": {
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2
    }
}

def load_advanced_defaults():
    """
    Load advanced parameter defaults from config file.
    Checks Eclipse folder first, falls back to repo defaults.
    Used for initializing widget default values.
    Note: Saving is handled by JavaScript via POST endpoint.
    """
    import folder_paths
    eclipse_config = os.path.join(folder_paths.models_dir, "Eclipse", "config", "smartlm_advanced_defaults.json")
    repo_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "config", "smartlm_advanced_defaults.json")
    
    # Check Eclipse first, then repo
    config_path = eclipse_config if os.path.exists(eclipse_config) else repo_config
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return DEFAULT_PARAMS
    except Exception as e:
        from ..core import cstr
        cstr(f"Could not load advanced defaults config: {e}").warning.print()
        return DEFAULT_PARAMS

# Load defaults on module initialization
ADVANCED_DEFAULTS = load_advanced_defaults()


class RvPipe_Out_smartlml_AdvancedOptions:
    # Advanced Smart LML Options Pipe - Configure advanced generation parameters
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["QwenVL", "Florence2", "LLM"], {"default": "QwenVL", "tooltip": "Select model type to show only relevant parameters. This doesn't change the model, just shows applicable settings."}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device to run model on (All models)"}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.compile for faster inference after warmup (QwenVL, Florence2)"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature for generation (QwenVL GGUF, LLM only)"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling parameter (QwenVL transformers, QwenVL GGUF, LLM)"}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "tooltip": "Top-k sampling - number of highest probability tokens to keep, 0 = disabled (QwenVL GGUF, LLM only)"}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Number of beams for beam search (All models)"}),
                "do_sample": ("BOOLEAN", {"default": True, "tooltip": "Use sampling instead of greedy decoding (All models)"}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalize repeated tokens, 1.0 = no penalty (QwenVL transformers, Florence2, QwenVL GGUF, LLM)"}),
                "frame_count": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Number of frames to sample from video (QwenVL only, ignored by Florence2/LLM)"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    RETURN_TYPES = ("SMARTLM_ADVANCED_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    
    OUTPUT_TOOLTIPS = ("Advanced options pipe for Smart LML nodes",)

    def execute(
        self,
        model_type,
        device,
        use_torch_compile,
        temperature,
        top_p,
        top_k,
        num_beams,
        do_sample,
        repetition_penalty,
        frame_count,
    ):
        # Package advanced options into a pipe dictionary and save user preferences
        # Note: model_type is for UI filtering only, all parameters are passed through
        
        pipe = {
            "device": device,
            "use_torch_compile": use_torch_compile,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "frame_count": frame_count,
        }
        
        # Note: Parameter saving is handled by JavaScript via POST endpoint
        # See eclipse-smart-loader-lm-advanced.js for auto-save implementation
        
        return (pipe,)

NODE_NAME = 'Pipe Out LM Advanced Options [Eclipse]'
NODE_DESC = 'Pipe Out Language Model Advanced Options'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_smartlml_AdvancedOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}