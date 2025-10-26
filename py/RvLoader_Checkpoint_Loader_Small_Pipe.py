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

import os
import comfy
import comfy.sd
import folder_paths

from ..core import CATEGORY, cstr

class RvLoader_Checkpoint_Loader_Small_Pipe:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), "Select the checkpoint file to load (v1 format)."),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"), "Select a VAE or choose 'Baked VAE' to use the one embedded in the checkpoint."),
                "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}, "Negative index to stop CLIP at. -1 means full CLIP."),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LOADER.value

    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    def execute(self, ckpt_name: str, vae_name: str, stop_at_clip_layer: int) -> tuple:
        # Input validation
        if not isinstance(ckpt_name, str) or not isinstance(vae_name, str):
            raise TypeError("ckpt_name and vae_name must be strings")
        if not isinstance(stop_at_clip_layer, int):
            raise TypeError("stop_at_clip_layer must be an int")

        try:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if not ckpt_path:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")

            # Moderate security: warn on absolute paths or parent traversal
            if os.path.isabs(ckpt_name):
                cstr("Warning: absolute checkpoint paths are discouraged").warning.print()
            if ".." in ckpt_name.replace('\\', '/'):
                cstr("Warning: parent-traversal sequences detected in checkpoint name").warning.print()

            # Resolve and verify checkpoint path
            ckpt_path_abs = os.path.abspath(ckpt_path)
            try:
                checkpoints_base = folder_paths.get_folder_paths("checkpoints")
                if isinstance(checkpoints_base, (list, tuple)):
                    base_candidate = checkpoints_base[0] if len(checkpoints_base) > 0 else None
                else:
                    base_candidate = checkpoints_base

                if base_candidate:
                    checkpoints_base_real = os.path.realpath(os.path.abspath(base_candidate))
                    ckpt_path_real = os.path.realpath(ckpt_path_abs)
                    try:
                        common = os.path.commonpath([checkpoints_base_real, ckpt_path_real])
                        if common != checkpoints_base_real:
                            cstr("Warning: resolved checkpoint path is outside the checkpoints folder").warning.print()
                    except Exception:
                        if not ckpt_path_abs.startswith(checkpoints_base_real):
                            cstr("Warning: resolved checkpoint path is outside the checkpoints folder").warning.print()
                else:
                    ckpt_path_real = os.path.realpath(ckpt_path_abs)
            except Exception:
                ckpt_path_real = os.path.realpath(ckpt_path_abs)

            # Ensure accessible
            if not os.path.isfile(ckpt_path_real) or not os.access(ckpt_path_real, os.R_OK):
                raise FileNotFoundError(f"Checkpoint file is not accessible: {ckpt_path_real}")

            # Extension handling: safetensors are considered safe; warn on legacy
            safe_exts = {".safetensors", ".sft"}
            legacy_exts = {".ckpt", ".pt", ".pth"}
            _, ext = os.path.splitext(ckpt_path_real.lower())
            if ext:
                if ext in legacy_exts:
                    cstr(f"Warning: legacy checkpoint extension detected: {ext}. Consider using .safetensors").warning.print()
                elif ext not in safe_exts:
                    cstr(f"Warning: unknown checkpoint extension: {ext}. Proceeding, but verify file.").warning.print()

            output_vae = (vae_name == "Baked VAE")

            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(
                ckpt_path_real,
                output_vae=output_vae,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )

            # Cache commonly accessed checkpoint parts to avoid repeated slicing
            ckpt_parts = loaded_ckpt[:3] if hasattr(loaded_ckpt, '__len__') and len(loaded_ckpt) >= 3 else None

            # VAE extraction
            if vae_name == "Baked VAE":
                loaded_vae = ckpt_parts[2] if ckpt_parts is not None else None
            else:
                vae_path = folder_paths.get_full_path("vae", vae_name)
                if not vae_path:
                    raise FileNotFoundError(f"VAE not found: {vae_name}")
                vae_path_real = os.path.realpath(os.path.abspath(vae_path))
                if not os.path.isfile(vae_path_real) or not os.access(vae_path_real, os.R_OK):
                    raise FileNotFoundError(f"VAE file is not accessible: {vae_path_real}")
                sd_obj = comfy.utils.load_torch_file(vae_path_real)
                loaded_vae = comfy.sd.VAE(sd=sd_obj)

            # CLIP: avoid cloning unless trimming is requested
            clip_candidate = ckpt_parts[1] if ckpt_parts is not None else getattr(loaded_ckpt, "clip", None)

            loaded_clip = None
            if clip_candidate is not None:
                if stop_at_clip_layer is not None and stop_at_clip_layer != -1:
                    try:
                        loaded_clip = clip_candidate.clone()
                        loaded_clip.clip_layer(stop_at_clip_layer)
                    except Exception:
                        loaded_clip = clip_candidate
                else:
                    loaded_clip = clip_candidate

            # Build canonical dict-style pipe
            model_obj = ckpt_parts[0] if ckpt_parts is not None and len(ckpt_parts) >= 1 else (loaded_ckpt if loaded_ckpt is not None else None)
            pipe = {
                "model": model_obj,
                "clip": loaded_clip,
                "vae": loaded_vae,
                "latent": {"samples": None},
                "width": None,
                "height": None,
                "batch_size": int(1),
                "model_name": ckpt_name,
                "vae_name": '' if vae_name == "Baked VAE" else str(vae_name),
                "clip_skip": int(stop_at_clip_layer),
            }
            return (pipe,)

        except Exception as e:
            cstr(f"Checkpoint loading failed: {e}").error.print()
            # Fail-fast: a missing or unreadable checkpoint should stop the graph
            # rather than returning a dummy pipe which can hide the root cause.
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e

NODE_NAME = 'Checkpoint Loader Small (Pipe) [RvTools]'
NODE_DESC = 'Checkpoint Loader Small (Pipe)'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLoader_Checkpoint_Loader_Small_Pipe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
