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

UNET_DOWNSAMPLE = 8

class RvPipe_Out_CheckpointLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input pipe containing model, clip, vae, latent, width, height, batch size, and names."}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Optional latent input to use if pipe does not supply latent, width, or height."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value

    # Output types with detailed descriptions
    RETURN_TYPES = (
        "MODEL",   # model: The UNet/diffusion model from checkpoint
        "CLIP",    # clip: Text encoder model for prompt conditioning
        "VAE",     # vae: Variational autoencoder for latent encoding/decoding
        "LATENT",  # latent: Latent tensor representation (noise or encoded image)
        "INT",     # steps: Number of sampling steps (Smart Loader Plus only)
        "FLOAT",   # cfg: Classifier-free guidance scale (Smart Loader Plus only)
        any_type,  # sampler: Sampling algorithm (ANY type - Smart Loader Plus only)
        any_type,  # scheduler: Noise reduction scheduler (ANY type - Smart Loader Plus only)
        "INT",     # clip_skip: Number of CLIP layers to skip (for style control)        
        "INT",     # width: Image width in pixels
        "INT",     # height: Image height in pixels
        "INT",     # batch_size: Number of images to generate simultaneously
        "STRING",  # model_name: Checkpoint filename/identifier
        "STRING",  # vae_name: VAE filename/identifier
        

    )
    
    RETURN_NAMES = (
        "model", 
        "clip", 
        "vae", 
        "latent", 
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "clip_skip",        
        "width", 
        "height", 
        "batch_size", 
        "model_name", 
        "vae_name", 
    )
    
    FUNCTION = "execute"

    def execute(self, pipe: Optional[dict[Any, Any]] = None, latent: Optional[dict[Any, Any]] = None) -> tuple:
        # Expect dict-style pipe. Tuples have been deprecated for pipe interchange.
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_CheckpointLoader expects dict-style pipes only.")

        # Prefer object fields; fall back to name fields only where appropriate.
        model = pipe.get("model")
        clip = pipe.get("clip")
        vae = pipe.get("vae")
        latent_from_pipe = pipe.get("latent")

        # Numeric fields and batch size
        width = pipe.get("width")
        height = pipe.get("height")
        batch_size = pipe.get("batch_size")
        clip_skip = pipe.get("clip_skip")

        # Coerce simple numeric types where appropriate
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
            if batch_size is not None:
                batch_size = int(batch_size)
        except Exception:
            batch_size = None

        # If pipe has no width/height and latent input is provided, derive from latent shape
        if (width is None or height is None) and latent is not None:
            latent_shape = latent["samples"].shape
            if height is None:
                height = latent_shape[2] * UNET_DOWNSAMPLE
            if width is None:
                width = latent_shape[3] * UNET_DOWNSAMPLE

        # Use latent from pipe if it has valid samples, otherwise fallback to input latent
        if latent_from_pipe is not None and latent_from_pipe.get("samples") is not None:
            output_latent = latent_from_pipe
        else:
            output_latent = latent

        # Name fields (textual)
        model_name = pipe.get("model_name")
        vae_name = pipe.get("vae_name")
        
        # Sampler settings (Smart Loader Plus only - defaults if not present)
        sampler = pipe.get("sampler_name")
        scheduler = pipe.get("scheduler")
        steps = pipe.get("steps", 20)
        cfg = pipe.get("cfg", 8.0)
        
        # Coerce sampler numeric types
        try:
            if steps is not None:
                steps = int(steps)
        except Exception:
            steps = 20
        try:
            if cfg is not None:
                cfg = float(cfg)
        except Exception:
            cfg = 8.0

        return (model, clip, vae, output_latent, steps, cfg, sampler, scheduler, clip_skip,width, height, batch_size, model_name, vae_name)

NODE_NAME = 'Pipe Out Checkpoint Loader [Eclipse]'
NODE_DESC = 'Pipe Out Checkpoint Loader'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_CheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}