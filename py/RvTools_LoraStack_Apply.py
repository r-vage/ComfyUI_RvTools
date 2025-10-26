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

import comfy
import folder_paths
from typing import Any
from ..core import CATEGORY, AnyType

any = AnyType("*")


def is_nunchaku_model(model: Any) -> bool:
    """
    Check if a model is a Nunchaku FLUX model by detecting ComfyFluxWrapper.
    
    Parameters
    ----------
    model : Any
        The model (ModelPatcher) to check.
        
    Returns
    -------
    bool
        True if the model has ComfyFluxWrapper, False otherwise.
    """
    try:
        model_wrapper = model.model.diffusion_model  # type: ignore
        
        # Check if it's a ComfyFluxWrapper (handle torch.compile() optimized modules)
        if hasattr(model_wrapper, '_orig_mod'):
            # This is a torch._dynamo.eval_frame.OptimizedModule
            actual_wrapper = model_wrapper._orig_mod  # type: ignore
            wrapper_class_name = type(actual_wrapper).__name__
            return wrapper_class_name == 'ComfyFluxWrapper'
        else:
            wrapper_class_name = type(model_wrapper).__name__
            return wrapper_class_name == 'ComfyFluxWrapper'
    except Exception:
        return False

class RvTools_LoraStack_Apply:


    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "lora_stack": ("LORA_STACK", ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_names")
    FUNCTION = "apply_lora_stack"
    
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def apply_lora_stack(self, model, clip, lora_stack=None,):
 
        # Initialise the list
        lora_params = list()
 
        # Extend lora_params with lora-stack items 
        if lora_stack:
            lora_params.extend(lora_stack)
        else:
            return (model, clip, "")

        # Check if this is a Nunchaku model
        if is_nunchaku_model(model):
            print("RvTools: [LoraStack Apply] Detected Nunchaku model, applying LoRAs via wrapper")
            return self._apply_lora_stack_nunchaku(model, clip, lora_params)
        else:
            # Standard model - use ComfyUI's load_lora_for_models
            return self._apply_lora_stack_standard(model, clip, lora_params)

    def _apply_lora_stack_standard(self, model, clip, lora_params):
        """Apply LoRAs to standard (non-Nunchaku) models using ComfyUI's loader."""
        # Initialise the model and clip
        model_lora = model
        clip_lora = clip

        # Loop through the list
        for tup in lora_params:
            lora_name, strength_model, strength_clip = tup
            
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_model, strength_clip)  

        # Generate string output with weights
        lora_string = ""
        if lora_params:
            try:
                # Format: <lora:name:model_weight:clip_weight>
                lora_string = ' '.join(
                    f"<lora:{str(tup[0])}:{str(tup[1])}:{str(tup[2])}>"
                    for tup in lora_params
                    if isinstance(tup, (list, tuple)) and len(tup) >= 3
                )
            except Exception:
                lora_string = ""

        return (model_lora, clip_lora, lora_string)

    def _apply_lora_stack_nunchaku(self, model: Any, clip: Any, lora_params: list[Any]) -> tuple[Any, Any, str]:
        """Apply LoRAs to Nunchaku FLUX models via ComfyFluxWrapper."""
        try:
            # Import required Nunchaku components
            from nunchaku.lora.flux import to_diffusers  # type: ignore
            from .wrappers.nunchaku_wrapper import ComfyFluxWrapper  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Nunchaku not available for LoRA application: {e}\n"
                "Please install ComfyUI-nunchaku extension."
            )

        # Get the model wrapper
        model_wrapper = model.model.diffusion_model  # type: ignore
        
        # Handle OptimizedModule case
        if hasattr(model_wrapper, '_orig_mod'):
            transformer = model_wrapper._orig_mod.model  # type: ignore
            
            # Create a new model structure manually for OptimizedModule
            ret_model = model.__class__(  # type: ignore
                model.model, model.load_device, model.offload_device,  # type: ignore
                model.size, model.weight_inplace_update  # type: ignore
            )
            ret_model.model = model.model  # type: ignore
            
            # Create a new ComfyFluxWrapper manually
            original_wrapper = model_wrapper._orig_mod  # type: ignore
            ret_model_wrapper = ComfyFluxWrapper(  # type: ignore
                transformer,
                original_wrapper.config,  # type: ignore
                original_wrapper.pulid_pipeline,  # type: ignore
                original_wrapper.customized_forward,  # type: ignore
                original_wrapper.forward_kwargs  # type: ignore
            )
            
            # Copy internal state from original wrapper
            ret_model_wrapper._prev_timestep = original_wrapper._prev_timestep  # type: ignore
            ret_model_wrapper._cache_context = original_wrapper._cache_context  # type: ignore
            if hasattr(original_wrapper, '_original_time_text_embed'):
                ret_model_wrapper._original_time_text_embed = original_wrapper._original_time_text_embed  # type: ignore
            
            ret_model.model.diffusion_model = ret_model_wrapper  # type: ignore
        else:
            # Non-OptimizedModule case
            transformer = model_wrapper.model  # type: ignore
            
            # Create a new ModelPatcher with the same parameters
            ret_model = model.__class__(  # type: ignore
                model.model, model.load_device, model.offload_device,  # type: ignore
                model.size, model.weight_inplace_update  # type: ignore
            )
            
            # Create a new ComfyFluxWrapper manually
            original_wrapper = model_wrapper
            ret_model_wrapper = ComfyFluxWrapper(  # type: ignore
                transformer,
                original_wrapper.config,  # type: ignore
                original_wrapper.pulid_pipeline,  # type: ignore
                original_wrapper.customized_forward,  # type: ignore
                original_wrapper.forward_kwargs  # type: ignore
            )
            
            # Copy internal state from original wrapper
            ret_model_wrapper._prev_timestep = original_wrapper._prev_timestep  # type: ignore
            ret_model_wrapper._cache_context = original_wrapper._cache_context  # type: ignore
            if hasattr(original_wrapper, '_original_time_text_embed'):
                ret_model_wrapper._original_time_text_embed = original_wrapper._original_time_text_embed  # type: ignore
            
            ret_model.model.diffusion_model = ret_model_wrapper  # type: ignore
        
        # Restore transformer to the original wrapper (important for original model integrity)
        if hasattr(model_wrapper, '_orig_mod'):
            model_wrapper._orig_mod.model = transformer  # type: ignore
        else:
            model_wrapper.model = transformer  # type: ignore
        
        # Set transformer to the new wrapper
        ret_model_wrapper.model = transformer  # type: ignore

        # Clear existing LoRA list in the new wrapper
        ret_model_wrapper.loras = []  # type: ignore

        # Track the maximum input channels needed
        max_in_channels = ret_model.model.model_config.unet_config["in_channels"]  # type: ignore

        # Add all LoRAs to the wrapper's LoRA list
        # lora_params format: [(lora_name, model_strength, clip_strength), ...]
        # For Nunchaku, we use model_strength as the LoRA strength
        lora_names_list = []
        for lora_name, model_strength, clip_strength in lora_params:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, model_strength))  # type: ignore
            lora_names_list.append(lora_name)

            # Check if input channels need to be updated
            sd = to_diffusers(lora_path)  # type: ignore
            if "transformer.x_embedder.lora_A.weight" in sd:
                new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
                assert new_in_channels % 4 == 0, f"Invalid LoRA input channels: {new_in_channels}"
                new_in_channels = new_in_channels // 4
                max_in_channels = max(max_in_channels, new_in_channels)

        # Update the model's input channels if needed
        if max_in_channels > ret_model.model.model_config.unet_config["in_channels"]:  # type: ignore
            ret_model.model.model_config.unet_config["in_channels"] = max_in_channels  # type: ignore

        # Generate string output with weights (Nunchaku uses model_strength only)
        lora_string = ""
        if lora_params:
            try:
                # Format: <lora:name:model_weight:clip_weight>
                lora_string = ' '.join(
                    f"<lora:{str(tup[0])}:{str(tup[1])}:{str(tup[2])}>"
                    for tup in lora_params
                    if isinstance(tup, (list, tuple)) and len(tup) >= 3
                )
            except Exception:
                lora_string = ""

        # For Nunchaku, CLIP is not modified (FLUX doesn't use separate CLIP)
        return (ret_model, clip, lora_string)

NODE_NAME = 'Lora Stack apply [RvTools]'
NODE_DESC = 'Lora Stack apply'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvTools_LoraStack_Apply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}