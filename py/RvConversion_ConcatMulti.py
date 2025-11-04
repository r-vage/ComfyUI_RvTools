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

from typing import Any, Dict
from ..core import CATEGORY

class RvConversion_ConcatMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 256, "step": 1}),
                "pipe_1": ("pipe",),
            },
            "optional": {
                "pipe_2": ("pipe",),
                "merge_strategy": (["overwrite", "preserve", "merge"], {
                    "default": "merge",
                    "tooltip": "How to handle conflicting keys:\n"
                              "'overwrite' replaces earlier values,\n"
                              "'preserve' keeps first valid values,\n"
                              "'merge' combines lists and uses later values for conflicts"
                }),
            },
        }

    RETURN_TYPES = ("pipe",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "concat"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.CONVERSION.value

    DESCRIPTION = "Merge multiple pipe/context inputs into a single context dict pipe."

    # keys that should be treated as list-like when merging
    _KNOWN_LIST_KEYS = {
        "images", 
        "images_ref", 
        "images_pp", 
        "images_pp1",
        "images_pp2", 
        "images_pp3",
        "mask", 
        "mask_1", 
        "mask_2", 
        "audio_in",
        "audio_out",
        "lora_names", 
        "loras", 
        "embeddings", 
        "positive_list", 
        "negative_list"
    }

    def _is_empty_value(self, value) -> bool:
        """Check if a value should be considered empty/invalid for merging"""
        if value is None:
            return True
        if isinstance(value, str):
            # Skip 'None' strings, empty strings, and whitespace-only strings
            return value.strip() in ('', 'None', 'none', 'null', 'NULL')
        if isinstance(value, (list, tuple)):
            # Skip empty collections
            return len(value) == 0
        return False

    def concat(self, inputcount: int = 2, merge_strategy: str = "merge", **kwargs) -> tuple:
        result: Dict[str, Any] = {}

        aliases = {
            "Steps": "steps",
            "CFG": "cfg",
            "model_name": "model_name",
            "lora_names": "lora_names",
            "loras": "lora_names",
            "seed": "seed",
            "sampler_name": "sampler_name",
            "sampler": "sampler_name",
            "vae_name": "vae_name",
            "directory": "path",
        }

        def set_value(k, v):
            # helper to set value into result respecting strategy
            if merge_strategy == "preserve":
                if k in result and not self._is_empty_value(result[k]):
                    return
                result[k] = v
                return

            if merge_strategy == "merge":
                if k in result:
                    existing = result[k]
                    # Special case for comma-separated strings in known list keys
                    if k in self._KNOWN_LIST_KEYS and isinstance(existing, str) and isinstance(v, str):
                        result[k] = existing + ", " + v
                        return
                    # if either is list/tuple, concatenate
                    if isinstance(existing, (list, tuple)) or isinstance(v, (list, tuple)) or k in self._KNOWN_LIST_KEYS:
                        existing_list = list(existing) if not isinstance(existing, list) else existing
                        new_list = list(v) if isinstance(v, (list, tuple)) else [v]
                        result[k] = existing_list + new_list
                        return
                # fallback to overwrite
                result[k] = v
                return

            # default overwrite
            result[k] = v

        for idx in range(1, inputcount + 1):
            pipe = kwargs.get(f"pipe_{idx}")
            if pipe is None:
                continue

            # Handle both dict-style pipes and tuple-style pipes (from context nodes)
            if isinstance(pipe, tuple):
                # Assume first element is the dict (like from context nodes)
                ctx = pipe[0] if pipe and isinstance(pipe[0], dict) else {}
            elif isinstance(pipe, dict):
                ctx = pipe
            else:
                raise ValueError(
                    f"Pipe input pipe_{idx} must be a dict or tuple containing a dict. Got: {type(pipe)}"
                )

            for k, v in ctx.items():
                # skip None/empty values when merging preserve/overwrite
                if self._is_empty_value(v):
                    continue

                # canonicalize common alias keys to a single repo-wide name
                key = aliases.get(k, k)

                # normalize simple list-like values to list when merge is used (but keep strings for concatenation)
                if merge_strategy == "merge" and key in self._KNOWN_LIST_KEYS and not isinstance(v, (list, tuple)) and not isinstance(v, str):
                    v = [v]

                set_value(key, v)

        # Ensure the result always contains a pipe key for convenience
        if "pipe" not in result:
            result["pipe"] = result

        return (result,)

NODE_NAME = 'Concat Pipe Multi [Eclipse]'
NODE_DESC = 'Concat Pipe Multi'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvConversion_ConcatMulti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}