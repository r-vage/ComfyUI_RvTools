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
import comfy

from ..core import CATEGORY, AnyType

any_type = AnyType("*")

class RvPipe_Out_LoadImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input pipe produced by Load Image (Metadata Pipe)"}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value

    # Outputs mirror the metadata fields produced by the Load Image (Metadata Pipe).
    # Note: image and mask tensor objects are not delivered through this pipe out.
    RETURN_TYPES = (
        "PIPE",    # original pipe dict
        "INT",     # width
        "INT",     # height
        "STRING",  # text_pos (positive prompt)
        "STRING",  # text_neg (negative prompt)
        "INT",     # steps
        "FLOAT",   # cfg
        any_type,  # sampler
        any_type,  # scheduler
        "INT",     # seed
        "STRING",  # model_name
        "STRING",  # path
    )

    RETURN_NAMES = (
        "pipe",
        "width",
        "height",
        "text_pos",
        "text_neg",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "seed",
        "model_name",
        "path",
    )

    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None):
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_LoadImage expects dict-style pipes only.")
        # Extract common fields with sensible defaults. image/mask are not forwarded.
        width = pipe.get("width")
        height = pipe.get("height")
        text_pos = pipe.get("text_pos") or pipe.get("text") or pipe.get("prompt") or ""
        text_neg = pipe.get("text_neg") or pipe.get("negative") or pipe.get("negative_prompt") or ""

        steps = pipe.get("steps", 0)
        sampler = pipe.get("sampler_name")
        scheduler = pipe.get("scheduler")
        cfg = pipe.get("cfg", 0.0)
        seed = pipe.get("seed", 0)

        model_name = pipe.get("model_name") or ""
        path = pipe.get("path") or ""

        # Coerce numeric types where reasonable
        try:
            if width is not None:
                width = int(width)
        except Exception:
            width = None
        try:
            if height is not None:
                height = int(height)
        except Exception:
            height = None
        try:
            steps = int(steps)
        except Exception:
            steps = 0
        try:
            cfg = float(cfg)
        except Exception:
            cfg = 0.0
        try:
            seed = int(seed)
        except Exception:
            seed = 0

        return (pipe, width, height, text_pos, text_neg, steps, cfg, sampler, scheduler, seed, model_name, path)


NODE_NAME = 'Pipe Out Load Image (Metadata Pipe) [RvTools]'
NODE_DESC = 'Pipe Out Load Image (Metadata Pipe)'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_LoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
