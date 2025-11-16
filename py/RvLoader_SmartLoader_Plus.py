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
from __future__ import annotations

"""
Smart Loader Plus - Advanced Model Loader with Integrated LoRA Support

Comprehensive model loader supporting multiple model formats and quantization methods:
- Standard Checkpoints (.safetensors, .ckpt)
- UNet-only models
- Nunchaku quantized models (Flux and Qwen-Image with SVDQuant INT4/FP4/FP8)
- GGUF quantized models (INT4/INT8 quantization)

Features:
- Automatic model type detection
- Format-specific loading options (cache, attention, offload)
- Template system for saving/loading configurations with intelligent field filtering
- Model-only LoRA support with up to 3 slots
- Graceful fallback when extensions are not installed
- Comprehensive VRAM management and cleanup
- Auto-fill template names for easy updates
"""

from typing import Any
import os
import sys
import json
import time
import gc

import torch
import comfy
import comfy.sd
import comfy.utils
import comfy.model_sampling
import folder_paths
import comfy.model_management as mm
import nodes

from ..core import CATEGORY, cstr, RESOLUTION_PRESETS, RESOLUTION_MAP
from comfy.comfy_types import IO

# Import Nunchaku wrapper
from .wrappers.nunchaku_wrapper import (
    NUNCHAKU_AVAILABLE,
    detect_nunchaku_model,
    load_nunchaku_model,
    get_nunchaku_info
)

# Import GGUF wrapper
from .wrappers.gguf_wrapper import (
    GGUF_AVAILABLE,
    detect_gguf_model,
    load_gguf_model
)

# Register custom folder path for GGUF diffusion models (if GGUF is available)
if GGUF_AVAILABLE:
    base = folder_paths.folder_names_and_paths.get("diffusion_models_gguf", ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    orig, _ = folder_paths.folder_names_and_paths.get("diffusion_models", ([], {}))
    folder_paths.folder_names_and_paths["diffusion_models_gguf"] = (orig or base, {".gguf"})
    
    # Add .gguf extension support to clip and text_encoders folders
    if "clip" in folder_paths.folder_names_and_paths:
        clip_data = folder_paths.folder_names_and_paths["clip"]
        clip_paths, clip_exts = clip_data[0], clip_data[1] if len(clip_data) >= 2 else ([], {})
        if ".gguf" not in clip_exts:
            clip_exts = set(clip_exts) if isinstance(clip_exts, set) else set(clip_exts.keys()) if isinstance(clip_exts, dict) else set()
            clip_exts.add(".gguf")
            folder_paths.folder_names_and_paths["clip"] = (clip_paths, clip_exts)
            # Clear cache to force re-scan with new extension
            if hasattr(folder_paths, 'filename_list_cache') and "clip" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["clip"]
            if hasattr(folder_paths, 'cache_helper'):
                folder_paths.cache_helper.clear()
    
    if "text_encoders" in folder_paths.folder_names_and_paths:
        te_data = folder_paths.folder_names_and_paths["text_encoders"]
        te_paths, te_exts = te_data[0], te_data[1] if len(te_data) >= 2 else ([], {})
        if ".gguf" not in te_exts:
            te_exts = set(te_exts) if isinstance(te_exts, set) else set(te_exts.keys()) if isinstance(te_exts, dict) else set()
            te_exts.add(".gguf")
            folder_paths.folder_names_and_paths["text_encoders"] = (te_paths, te_exts)
            # Clear cache to force re-scan with new extension
            if hasattr(folder_paths, 'filename_list_cache') and "text_encoders" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["text_encoders"]
            if hasattr(folder_paths, 'cache_helper'):
                folder_paths.cache_helper.clear()

# Add .safetensors and .sft extension support to checkpoints and diffusion_models folders
for folder_name in ["checkpoints", "diffusion_models"]:
    if folder_name in folder_paths.folder_names_and_paths:
        folder_data = folder_paths.folder_names_and_paths[folder_name]
        # Handle both 2-tuple and 3-tuple formats
        if len(folder_data) >= 2:
            paths, exts = folder_data[0], folder_data[1]
        else:
            continue
        exts = set(exts) if isinstance(exts, set) else set(exts.keys()) if isinstance(exts, dict) else set()
        # Ensure common extensions are present
        for ext in [".safetensors", ".sft", ".ckpt", ".pt", ".bin"]:
            exts.add(ext)
        folder_paths.folder_names_and_paths[folder_name] = (paths, exts)

MAX_RESOLUTION = 32768
LATENT_CHANNELS = 4
UNET_DOWNSAMPLE = 8

# Template system - use ComfyUI models folder as primary location
TEMPLATE_DIR = os.path.join(folder_paths.models_dir, "smart_loader_templates")

def cleanup_memory_before_load():
    """Clean up memory before loading a new model."""
    cstr("[Memory Cleanup] Starting pre-load memory cleanup...").msg.print()
    gc.collect()
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        cstr(f"[Memory Cleanup] Clearing CUDA cache on {device_count} device(s)").msg.print()
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    if hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'):
        try:
            torch.mps.empty_cache()
            cstr("[Memory Cleanup] Cleared MPS cache").msg.print()
        except Exception:
            pass
    
    if hasattr(mm, 'soft_empty_cache'):
        mm.soft_empty_cache()
    
    cstr("[Memory Cleanup] ✓ Memory cleanup complete").msg.print()

def ensure_template_dir():
    """Ensure template directory exists"""
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

def get_template_list():
    """Get list of available templates"""
    ensure_template_dir()
    templates = ["None"]
    try:
        for f in os.listdir(TEMPLATE_DIR):
            if f.endswith('.json'):
                templates.append(f[:-5])
    except Exception:
        pass
    return templates

def save_template(name: str, config: dict):
    """Save a configuration template"""
    ensure_template_dir()
    template_path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    try:
        with open(template_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        cstr(f"Error saving template: {e}").error.print()
        return False

def load_template(name: str) -> dict:
    """Load a configuration template"""
    if name == "None" or not name:
        return {}
    
    template_path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    try:
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        cstr(f"Error loading template: {e}").error.print()
    return {}

def delete_template(name: str):
    """Delete a configuration template"""
    if name == "None" or not name:
        return False
    
    template_path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    try:
        if os.path.exists(template_path):
            os.remove(template_path)
            return True
    except Exception as e:
        cstr(f"Error deleting template: {e}").error.print()
    return False

def _detect_latent_channels_from_vae_obj(vae_obj) -> int:
    """Infer latent channel count from a VAE-like object."""
    try:
        if hasattr(vae_obj, 'channels') and isinstance(getattr(vae_obj, 'channels'), int):
            return getattr(vae_obj, 'channels')
        if hasattr(vae_obj, 'latent_channels') and isinstance(getattr(vae_obj, 'latent_channels'), int):
            return getattr(vae_obj, 'latent_channels')
        for attr in ('encoder', 'conv_in', 'down_blocks'):
            sub = getattr(vae_obj, attr, None)
            if sub is not None and hasattr(sub, 'weight'):
                return sub.weight.shape[0]
    except Exception:
        pass
    return LATENT_CHANNELS

def is_nunchaku_model(model: Any) -> bool:
    """Check if a model is a Nunchaku FLUX model by detecting ComfyFluxWrapper."""
    try:
        model_wrapper = model.model.diffusion_model
        
        if hasattr(model_wrapper, '_orig_mod'):
            actual_wrapper = model_wrapper._orig_mod
            wrapper_class_name = type(actual_wrapper).__name__
            return wrapper_class_name == 'ComfyFluxWrapper'
        else:
            wrapper_class_name = type(model_wrapper).__name__
            return wrapper_class_name == 'ComfyFluxWrapper'
    except Exception:
        return False

def apply_loras_to_model(model: Any, clip: Any, lora_params: list) -> tuple:
    """
    Apply LoRAs to model (standard or Nunchaku).
    
    Parameters:
        model: The model to apply LoRAs to
        clip: The CLIP model (for standard models only)
        lora_params: List of tuples (lora_name, model_weight)
    
    Returns:
        (modified_model, modified_clip)
    """
    if not lora_params:
        return (model, clip)
    
    # Check if this is a Nunchaku model
    if is_nunchaku_model(model):
        cstr("[LoRA] Detected Nunchaku model, applying LoRAs via wrapper").msg.print()
        return _apply_loras_nunchaku(model, clip, lora_params)
    else:
        cstr("[LoRA] Applying LoRAs to standard model").msg.print()
        return _apply_loras_standard(model, clip, lora_params)

def _apply_loras_standard(model: Any, clip: Any, lora_params: list) -> tuple:
    """Apply LoRAs to standard (non-Nunchaku) models using ComfyUI's loader."""
    model_lora = model
    clip_lora = clip
    
    for lora_name, model_weight in lora_params:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # Use model_weight for both model and clip (model-only mode)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model_lora, clip_lora, lora, model_weight, model_weight
        )
        cstr(f"[LoRA] Applied {lora_name} with weight {model_weight}").msg.print()
    
    return (model_lora, clip_lora)

def _apply_loras_nunchaku(model: Any, clip: Any, lora_params: list) -> tuple:
    """Apply LoRAs to Nunchaku FLUX models via ComfyFluxWrapper."""
    try:
        from nunchaku.lora.flux import to_diffusers
        from .wrappers.nunchaku_wrapper import ComfyFluxWrapper
    except ImportError as e:
        cstr(f"[LoRA] Nunchaku not available for LoRA application: {e}").warning.print()
        cstr("[LoRA] Returning model unchanged").msg.print()
        return (model, clip)
    
    if ComfyFluxWrapper is None:
        cstr("[LoRA] ComfyFluxWrapper not available - returning model unchanged").warning.print()
        return (model, clip)
    
    # Get the model wrapper
    model_wrapper = model.model.diffusion_model
    
    # Handle OptimizedModule case
    if hasattr(model_wrapper, '_orig_mod'):
        transformer = model_wrapper._orig_mod.model
        
        # Create a new model structure
        ret_model = model.__class__(
            model.model, model.load_device, model.offload_device,
            model.size, model.weight_inplace_update
        )
        ret_model.model = model.model
        
        # Create a new ComfyFluxWrapper
        original_wrapper = model_wrapper._orig_mod
        ret_model_wrapper = ComfyFluxWrapper(
            transformer,
            original_wrapper.config,
            original_wrapper.pulid_pipeline,
            original_wrapper.customized_forward,
            original_wrapper.forward_kwargs
        )
        
        # Copy internal state
        ret_model_wrapper._prev_timestep = original_wrapper._prev_timestep
        ret_model_wrapper._cache_context = original_wrapper._cache_context
        if hasattr(original_wrapper, '_original_time_text_embed'):
            ret_model_wrapper._original_time_text_embed = original_wrapper._original_time_text_embed
        
        ret_model.model.diffusion_model = ret_model_wrapper
    else:
        # Non-OptimizedModule case
        transformer = model_wrapper.model
        
        ret_model = model.__class__(
            model.model, model.load_device, model.offload_device,
            model.size, model.weight_inplace_update
        )
        
        original_wrapper = model_wrapper
        ret_model_wrapper = ComfyFluxWrapper(
            transformer,
            original_wrapper.config,
            original_wrapper.pulid_pipeline,
            original_wrapper.customized_forward,
            original_wrapper.forward_kwargs
        )
        
        # Copy internal state
        ret_model_wrapper._prev_timestep = original_wrapper._prev_timestep
        ret_model_wrapper._cache_context = original_wrapper._cache_context
        if hasattr(original_wrapper, '_original_time_text_embed'):
            ret_model_wrapper._original_time_text_embed = original_wrapper._original_time_text_embed
        
        ret_model.model.diffusion_model = ret_model_wrapper
    
    # Restore transformer to original wrapper
    if hasattr(model_wrapper, '_orig_mod'):
        model_wrapper._orig_mod.model = transformer
    else:
        model_wrapper.model = transformer
    
    ret_model_wrapper.model = transformer
    
    # Clear existing LoRA list
    ret_model_wrapper.loras = []
    
    # Track max input channels
    max_in_channels = ret_model.model.model_config.unet_config["in_channels"]
    
    # Add all LoRAs
    for lora_name, model_weight in lora_params:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, model_weight))
        cstr(f"[LoRA] Applied Nunchaku LoRA {lora_name} with weight {model_weight}").msg.print()
        
        # Check input channels
        sd = to_diffusers(lora_path)
        if "transformer.x_embedder.lora_A.weight" in sd:
            new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
            assert new_in_channels % 4 == 0, f"Invalid LoRA input channels: {new_in_channels}"
            new_in_channels = new_in_channels // 4
            max_in_channels = max(max_in_channels, new_in_channels)
    
    # Update input channels if needed
    if max_in_channels > ret_model.model.model_config.unet_config["in_channels"]:
        ret_model.model.model_config.unet_config["in_channels"] = max_in_channels
    
    return (ret_model, clip)

def apply_model_sampling(model, sampling_method: str, shift: float, base_shift: float = 0.5, 
                         width: int = 1024, height: int = 1024, original_timesteps: int = 50,
                         zsnr: bool = False, sampling_subtype: str = "eps", 
                         sigma_max: float = 120.0, sigma_min: float = 0.002):
    """
    Apply model sampling configuration based on method.
    
    Parameters:
        model: The model to patch
        sampling_method: Sampling method (SD3, AuraFlow, Flux, Stable Cascade, LCM, ContinuousEDM, ContinuousV, LTXV, or None)
        shift: Universal shift parameter (used as shift for SD3/AuraFlow/Stable Cascade, max_shift for Flux/LTXV)
        base_shift: Base shift for Flux/LTXV sampling (default: 0.5)
        width: Width for Flux sampling shift calculation (default: 1024)
        height: Height for Flux sampling shift calculation (default: 1024)
        original_timesteps: Original timesteps for LCM sampling (default: 50)
        zsnr: Enable zero-terminal SNR for LCM sampling (default: False)
        sampling_subtype: Subtype for ContinuousEDM (eps, v_prediction, edm, edm_playground_v2.5, cosmos_rflow)
        sigma_max: Maximum sigma for ContinuousEDM/V (default: 120.0)
        sigma_min: Minimum sigma for ContinuousEDM/V (default: 0.002)
    
    Returns:
        Patched model or original model if method is "None"
    """
    if sampling_method == "None" or not sampling_method:
        return model
    
    if sampling_method == "SD3":
        return _apply_sd3_sampling(model, shift=shift, multiplier=1000.0)
    elif sampling_method == "AuraFlow":
        return _apply_auraflow_sampling(model, shift=shift, multiplier=1.0)
    elif sampling_method == "Flux":
        return _apply_flux_sampling(model, max_shift=shift, base_shift=base_shift, width=width, height=height)
    elif sampling_method == "Stable Cascade":
        return _apply_stable_cascade_sampling(model, shift=shift)
    elif sampling_method == "LCM":
        return _apply_lcm_sampling(model, original_timesteps=original_timesteps, zsnr=zsnr)
    elif sampling_method == "ContinuousEDM":
        return _apply_continuous_edm_sampling(model, sampling_subtype=sampling_subtype, sigma_max=sigma_max, sigma_min=sigma_min)
    elif sampling_method == "ContinuousV":
        return _apply_continuous_v_sampling(model, sigma_max=sigma_max, sigma_min=sigma_min)
    elif sampling_method == "LTXV":
        return _apply_ltxv_sampling(model, max_shift=shift, base_shift=base_shift)
    else:
        cstr(f"[Model Sampling] Unknown sampling method '{sampling_method}', skipping").warning.print()
        return model

def _apply_sd3_sampling(model, shift: float, multiplier: float = 1000.0):
    """Apply SD3 sampling (ModelSamplingDiscreteFlow + CONST)"""
    m = model.clone()
    
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=multiplier)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied SD3 sampling: shift={shift}, multiplier={multiplier}").msg.print()
    return m

def _apply_auraflow_sampling(model, shift: float, multiplier: float = 1.0):
    """Apply AuraFlow sampling (ModelSamplingDiscreteFlow + CONST with multiplier=1.0)"""
    m = model.clone()
    
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=multiplier)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied AuraFlow sampling: shift={shift}, multiplier={multiplier}").msg.print()
    return m

def _apply_flux_sampling(model, max_shift: float, base_shift: float, width: int, height: int):
    """Apply Flux sampling (ModelSamplingFlux + CONST with calculated shift)"""
    m = model.clone()
    
    # Calculate shift using linear interpolation formula based on image dimensions
    # Formula: shift = ((width * height / 1024) * slope) + intercept
    # where slope = (max_shift - base_shift) / (4096 - 256)
    # and intercept = base_shift - slope * 256
    x1 = 256
    x2 = 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    shift = (width * height / (8 * 8 * 2 * 2)) * mm + b
    
    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = comfy.model_sampling.CONST
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied Flux sampling: max_shift={max_shift}, base_shift={base_shift}, width={width}, height={height}, calculated_shift={shift:.4f}").msg.print()
    return m

def _apply_stable_cascade_sampling(model, shift: float):
    """Apply Stable Cascade sampling (StableCascadeSampling + EPS)"""
    m = model.clone()
    
    sampling_base = comfy.model_sampling.StableCascadeSampling
    sampling_type = comfy.model_sampling.EPS
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied Stable Cascade sampling: shift={shift}").msg.print()
    return m

def _apply_lcm_sampling(model, original_timesteps: int = 50, zsnr: bool = False):
    """Apply LCM sampling (ModelSamplingDiscreteDistilled + LCM)"""
    m = model.clone()
    
    # Define LCM sampling type
    class LCM(comfy.model_sampling.EPS):
        def calculate_denoised(self, sigma, model_output, model_input):
            timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
            sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
            x0 = model_input - model_output * sigma

            sigma_data = 0.5
            scaled_timestep = timestep * 10.0  # timestep_scaling

            c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
            c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

            return c_out * x0 + c_skip * model_input
    
    # Define distilled sampling base
    class ModelSamplingDiscreteDistilled(comfy.model_sampling.ModelSamplingDiscrete):
        def __init__(self, model_config=None):
            super().__init__(model_config, zsnr=zsnr)
            self.original_timesteps = original_timesteps
            self.skip_steps = self.num_timesteps // self.original_timesteps

            sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
            for x in range(self.original_timesteps):
                sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

            self.set_sigmas(sigmas_valid)

        def timestep(self, sigma):
            log_sigma = sigma.log()
            dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
            return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

        def sigma(self, timestep):
            t = torch.clamp(((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
            low_idx = t.floor().long()
            high_idx = t.ceil().long()
            w = t.frac()
            log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
            return log_sigma.exp().to(timestep.device)
    
    sampling_base = ModelSamplingDiscreteDistilled
    sampling_type = LCM
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied LCM sampling: original_timesteps={original_timesteps}, zsnr={zsnr}").msg.print()
    return m

def _apply_continuous_edm_sampling(model, sampling_subtype: str = "eps", sigma_max: float = 120.0, sigma_min: float = 0.002):
    """Apply ContinuousEDM sampling"""
    m = model.clone()
    
    sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
    latent_format = None
    sigma_data = 1.0
    
    if sampling_subtype == "eps":
        sampling_type = comfy.model_sampling.EPS
    elif sampling_subtype == "edm" or sampling_subtype == "edm_playground_v2.5":
        sampling_type = comfy.model_sampling.EDM
        sigma_data = 0.5
        if sampling_subtype == "edm_playground_v2.5":
            latent_format = comfy.latent_formats.SDXL_Playground_2_5()
    elif sampling_subtype == "v_prediction":
        sampling_type = comfy.model_sampling.V_PREDICTION
    elif sampling_subtype == "cosmos_rflow":
        sampling_type = comfy.model_sampling.COSMOS_RFLOW
        sampling_base = comfy.model_sampling.ModelSamplingCosmosRFlow
    else:
        cstr(f"[Model Sampling] Unknown ContinuousEDM subtype '{sampling_subtype}', using eps").warning.print()
        sampling_type = comfy.model_sampling.EPS
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
    m.add_object_patch("model_sampling", model_sampling)
    if latent_format is not None:
        m.add_object_patch("latent_format", latent_format)
    
    cstr(f"[Model Sampling] Applied ContinuousEDM sampling: subtype={sampling_subtype}, sigma_max={sigma_max}, sigma_min={sigma_min}, sigma_data={sigma_data}").msg.print()
    return m

def _apply_continuous_v_sampling(model, sigma_max: float = 500.0, sigma_min: float = 0.03):
    """Apply ContinuousV sampling (v_prediction only)"""
    m = model.clone()
    
    sampling_type = comfy.model_sampling.V_PREDICTION
    sigma_data = 1.0
    
    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousV, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied ContinuousV sampling: sigma_max={sigma_max}, sigma_min={sigma_min}").msg.print()
    return m

def _apply_ltxv_sampling(model, max_shift: float = 2.05, base_shift: float = 0.95):
    """Apply LTXV sampling (for video models, uses token-based shift calculation)"""
    m = model.clone()
    
    # LTXV uses token count instead of width/height
    # Default to 4096 tokens if we can't determine from latent
    tokens = 4096
    
    # Calculate shift using linear interpolation formula based on token count
    x1 = 1024
    x2 = 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    shift = tokens * mm + b
    
    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = comfy.model_sampling.CONST
    
    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass
    
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", model_sampling)
    
    cstr(f"[Model Sampling] Applied LTXV sampling: max_shift={max_shift}, base_shift={base_shift}, tokens={tokens}, calculated_shift={shift:.4f}").msg.print()
    return m

_support_messages_printed = False

class RvLoader_SmartLoader_Plus:
    resolution = RESOLUTION_PRESETS
    resolution_map = RESOLUTION_MAP
    
    def __init__(self):
        global _support_messages_printed
        if not _support_messages_printed:
            _support_messages_printed = True
            
            nunchaku_info = get_nunchaku_info()
            if nunchaku_info['available']:
                version = nunchaku_info['version'] if nunchaku_info['version'] else 'installed'
                cstr(f"✓ Nunchaku support: {version}").msg.print()
            
            if GGUF_AVAILABLE:
                cstr("✓ GGUF support available").msg.print()

    @classmethod
    def INPUT_TYPES(cls):
        nunchaku_info = get_nunchaku_info()
        weight_dtype_options = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
        
        # Get available LoRAs
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        # Get available CLIP files from both clip and text_encoders folders
        clip_files = []
        # Get from clip folder
        clip_files.extend(folder_paths.get_filename_list("clip"))
        # Get from text_encoders folder if it exists
        if "text_encoders" in folder_paths.folder_names_and_paths:
            clip_files.extend(folder_paths.get_filename_list("text_encoders"))
        clips = ["None"] + clip_files
        
        inputs = {
            "required": {
                # Template management
                "template_action": (["None", "Load", "Save", "Delete"], {
                    "default": "None",
                    "tooltip": "Load/Save/Delete configuration templates"
                }),
                "template_name": (get_template_list(), {
                    "default": "None",
                    "tooltip": "Select template to load/delete"
                }),
                "new_template_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name for new template (when saving)"
                }),
                
                # Model selection
                "model_type": (["Standard Checkpoint", "UNet Model", "Nunchaku Flux", "Nunchaku Qwen", "GGUF Model"], {
                    "default": "Standard Checkpoint",
                    "tooltip": "Select model type"
                }),
                "ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), {
                    "default": "None",
                    "tooltip": "Select checkpoint file"
                }),
                "unet_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select UNet diffusion model"
                }),
                "nunchaku_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select Nunchaku Flux model"
                }),
                "qwen_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select Nunchaku Qwen model"
                }),
                "gguf_name": (["None"] + (folder_paths.get_filename_list("diffusion_models_gguf") if "diffusion_models_gguf" in folder_paths.folder_names_and_paths else []), {
                    "default": "None",
                    "tooltip": "Select GGUF model"
                }),
                "weight_dtype": (weight_dtype_options, {
                    "default": "default",
                    "tooltip": "Weight dtype for UNet model"
                }),
                
                # Nunchaku settings
                "data_type": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                    "tooltip": "Model data type for Nunchaku"
                }),
                "cache_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Cache threshold for Nunchaku"
                }),
                "attention": (["flash-attention2", "nunchaku-fp16"], {
                    "default": "flash-attention2",
                    "tooltip": "Attention implementation"
                }),
                "i2f_mode": (["enabled", "always"], {
                    "default": "enabled",
                    "tooltip": "GEMM implementation"
                }),
                "cpu_offload": (["auto", "enable", "disable"], {
                    "default": "auto",
                    "tooltip": "CPU offload"
                }),
                "num_blocks_on_gpu": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Blocks on GPU (Nunchaku Qwen)"
                }),
                "use_pin_memory": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Use pinned memory"
                }),
                
                # GGUF settings
                "gguf_dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "Dequantization dtype"
                }),
                "gguf_patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "LoRA patch dtype"
                }),
                "gguf_patch_on_device": ("BOOLEAN", {
                    "default": False, "label_on": "yes", "label_off": "no",
                    "tooltip": "Apply patches on GPU"
                }),
                
                # Configuration toggles
                "configure_clip": ("BOOLEAN", {
                    "default": True, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable CLIP configuration"
                }),
                "configure_vae": ("BOOLEAN", {
                    "default": True, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable VAE configuration"
                }),
                "configure_latent": ("BOOLEAN", {
                    "default": True, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable latent configuration"
                }),
                "configure_sampler": ("BOOLEAN", {
                    "default": True, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable sampler configuration"
                }),
                "configure_model_only_lora": ("BOOLEAN", {
                    "default": False, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable model-only LoRA configuration"
                }),
                "configure_model_sampling": ("BOOLEAN", {
                    "default": False, "label_on": "yes", "label_off": "no",
                    "tooltip": "Enable advanced model sampling configuration"
                }),
                
                # Model Sampling Configuration
                "sampling_method": (["None", "SD3", "AuraFlow", "Flux", "Stable Cascade", "LCM", "ContinuousEDM", "ContinuousV", "LTXV"], {
                    "default": "None",
                    "tooltip": "Sampling method: SD3 (shift=3.0), AuraFlow (shift=1.73), Flux (max_shift=1.15), Stable Cascade (shift=2.0), LCM (distilled), ContinuousEDM/V (continuous sampling), LTXV (video)"
                }),
                "sampling_subtype": (["eps", "v_prediction", "edm", "edm_playground_v2.5", "cosmos_rflow"], {
                    "default": "eps",
                    "tooltip": "Subtype for ContinuousEDM sampling (eps, v_prediction, edm, edm_playground_v2.5, cosmos_rflow)"
                }),
                "shift": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Universal shift parameter (SD3: 3.0, AuraFlow: 1.73, Flux max_shift: 1.15, Stable Cascade: 2.0)"
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Base shift for Flux/LTXV sampling (default: 0.5)"
                }),
                "sampling_width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Width for Flux sampling shift calculation"
                }),
                "sampling_height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Height for Flux sampling shift calculation"
                }),
                "original_timesteps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Original timesteps for LCM sampling (default: 50)"
                }),
                "zsnr": ("BOOLEAN", {
                    "default": False,
                    "label_on": "yes",
                    "label_off": "no",
                    "tooltip": "Enable zero-terminal SNR for LCM sampling"
                }),
                "sigma_max": ("FLOAT", {
                    "default": 120.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Maximum sigma for ContinuousEDM/V sampling (EDM: 120.0, V: 500.0)"
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.002,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Minimum sigma for ContinuousEDM/V sampling (EDM: 0.002, V: 0.03)"
                }),
                
                # CLIP Configuration Section
                "clip_source": (["Baked", "External"], {
                    "default": "Baked",
                    "tooltip": "CLIP source"
                }),
                "clip_count": (["1", "2", "3", "4"], {
                    "default": "1",
                    "tooltip": "Number of CLIP models"
                }),
                "clip_name1": (clips, {
                    "default": "None",
                    "tooltip": "Primary CLIP model"
                }),
                "clip_name2": (clips, {
                    "default": "None",
                    "tooltip": "Secondary CLIP model"
                }),
                "clip_name3": (clips, {
                    "default": "None",
                    "tooltip": "Third CLIP model"
                }),
                "clip_name4": (clips, {
                    "default": "None",
                    "tooltip": "Fourth CLIP model"
                }),
                "clip_type": (["flux", "sd3", "sdxl", "stable_cascade", "stable_audio", "hunyuan_dit", "mochi", "ltxv", "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], {
                    "default": "flux",
                    "tooltip": "CLIP architecture type"
                }),
                "enable_clip_layer": ("BOOLEAN", {
                    "default": True,
                    "label_on": "yes",
                    "label_off": "no",
                    "tooltip": "Trim CLIP to specific layer"
                }),
                "stop_at_clip_layer": ("INT", {
                    "default": -2,
                    "min": -24,
                    "max": -1,
                    "step": 1,
                    "tooltip": "CLIP layer to stop at"
                }),
                
                # VAE settings
                "vae_source": (["Baked", "External"], {
                    "default": "Baked",
                    "tooltip": "VAE source"
                }),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"), {
                    "default": "None",
                    "tooltip": "External VAE file"
                }),
                
                # Latent settings
                "resolution": (cls.resolution, {
                    "default": "1024x1024 (1:1)",
                    "tooltip": "Preset resolution or Custom"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Custom width"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Custom height"
                }),
                
                # LoRA configuration
                "lora_count": (["1", "2", "3"], {
                    "default": "1",
                    "tooltip": "Number of LoRA slots to configure"
                }),
                
                # LoRA slot 1
                "lora_switch_1": ("BOOLEAN", {
                    "default": False, "label_on": "ON", "label_off": "OFF",
                    "tooltip": "Enable LoRA 1"
                }),
                "lora_name_1": (loras, {
                    "default": "None",
                    "tooltip": "LoRA 1 file"
                }),
                "lora_weight_1": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA 1 model weight"
                }),
                
                # LoRA slot 2
                "lora_switch_2": ("BOOLEAN", {
                    "default": False, "label_on": "ON", "label_off": "OFF",
                    "tooltip": "Enable LoRA 2"
                }),
                "lora_name_2": (loras, {
                    "default": "None",
                    "tooltip": "LoRA 2 file"
                }),
                "lora_weight_2": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA 2 model weight"
                }),
                
                # LoRA slot 3
                "lora_switch_3": ("BOOLEAN", {
                    "default": False, "label_on": "ON", "label_off": "OFF",
                    "tooltip": "Enable LoRA 3"
                }),
                "lora_name_3": (loras, {
                    "default": "None",
                    "tooltip": "LoRA 3 file"
                }),
                "lora_weight_3": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA 3 model weight"
                }),
                
                # Sampler settings
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampling algorithm"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Scheduler"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "CFG scale"
                }),
                "flux_guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Flux guidance scale"
                }),
                
                # Latent batch size
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Batch size"
                }),
                
                # Memory management
                "memory_cleanup": ("BOOLEAN", {
                    "default": True, "label_on": "yes", "label_off": "no",
                    "tooltip": "Perform memory cleanup before loading"
                }),
            },
        }
        
        return inputs

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LOADER.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        template_dir = TEMPLATE_DIR
        if os.path.exists(template_dir):
            try:
                return str(max(os.path.getmtime(os.path.join(template_dir, f)) 
                          for f in os.listdir(template_dir) if f.endswith('.json')))
            except (ValueError, OSError):
                pass
        return str(time.time())

    def execute(self, **kwargs):
        # Extract all parameters
        template_action = kwargs.get('template_action', 'None')
        template_name = kwargs.get('template_name', 'None')
        new_template_name = kwargs.get('new_template_name', '')
        
        model_type = kwargs.get('model_type', 'Standard Checkpoint')
        ckpt_name = kwargs.get('ckpt_name', 'None')
        unet_name = kwargs.get('unet_name', 'None')
        nunchaku_name = kwargs.get('nunchaku_name', 'None')
        qwen_name = kwargs.get('qwen_name', 'None')
        gguf_name = kwargs.get('gguf_name', 'None')
        weight_dtype = kwargs.get('weight_dtype', 'default')
        
        data_type = kwargs.get('data_type', 'bfloat16')
        cache_threshold = kwargs.get('cache_threshold', 0.0)
        attention = kwargs.get('attention', 'flash-attention2')
        i2f_mode = kwargs.get('i2f_mode', 'enabled')
        cpu_offload = kwargs.get('cpu_offload', 'auto')
        num_blocks_on_gpu = kwargs.get('num_blocks_on_gpu', 30)
        use_pin_memory = kwargs.get('use_pin_memory', 'enable')
        
        gguf_dequant_dtype = kwargs.get('gguf_dequant_dtype', 'default')
        gguf_patch_dtype = kwargs.get('gguf_patch_dtype', 'default')
        gguf_patch_on_device = kwargs.get('gguf_patch_on_device', False)
        
        configure_clip = kwargs.get('configure_clip', True)
        configure_vae = kwargs.get('configure_vae', True)
        configure_latent = kwargs.get('configure_latent', True)
        configure_sampler = kwargs.get('configure_sampler', True)
        configure_model_only_lora = kwargs.get('configure_model_only_lora', False)
        configure_model_sampling = kwargs.get('configure_model_sampling', False)
        
        sampling_method = kwargs.get('sampling_method', 'None')
        sampling_subtype = kwargs.get('sampling_subtype', 'eps')
        shift = kwargs.get('shift', 3.0)
        base_shift = kwargs.get('base_shift', 0.5)
        sampling_width = kwargs.get('sampling_width', 1024)
        sampling_height = kwargs.get('sampling_height', 1024)
        original_timesteps = kwargs.get('original_timesteps', 50)
        zsnr = kwargs.get('zsnr', False)
        sigma_max = kwargs.get('sigma_max', 120.0)
        sigma_min = kwargs.get('sigma_min', 0.002)
        
        clip_source = kwargs.get('clip_source', 'Baked')
        clip_count = kwargs.get('clip_count', '1')
        clip_name1 = kwargs.get('clip_name1', 'None')
        clip_name2 = kwargs.get('clip_name2', 'None')
        clip_name3 = kwargs.get('clip_name3', 'None')
        clip_name4 = kwargs.get('clip_name4', 'None')
        clip_type = kwargs.get('clip_type', 'flux')
        enable_clip_layer = kwargs.get('enable_clip_layer', True)
        stop_at_clip_layer = kwargs.get('stop_at_clip_layer', -2)
        
        vae_source = kwargs.get('vae_source', 'Baked')
        vae_name = kwargs.get('vae_name', 'None')
        
        resolution = kwargs.get('resolution', '1024x1024 (1:1)')
        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 1024)
        batch_size = kwargs.get('batch_size', 1)
        
        lora_count = kwargs.get('lora_count', '1')
        
        sampler_name = kwargs.get('sampler_name', 'euler')
        scheduler = kwargs.get('scheduler', 'normal')
        steps = kwargs.get('steps', 20)
        cfg = kwargs.get('cfg', 8.0)
        flux_guidance = kwargs.get('flux_guidance', 3.5)
        
        memory_cleanup = kwargs.get('memory_cleanup', True)
        
        # Handle template actions - Save and Delete interrupt, Load doesn't
        if template_action == "Save":
            if new_template_name and new_template_name.strip():
                config = {
                    "model_type": model_type,
                    "configure_clip": configure_clip,
                    "configure_vae": configure_vae,
                    "configure_latent": configure_latent,
                    "configure_sampler": configure_sampler,
                    "configure_model_only_lora": configure_model_only_lora,
                    "configure_model_sampling": configure_model_sampling,
                }
                
                # Only save the model field that matches the model_type
                if model_type == "Standard Checkpoint":
                    if ckpt_name != "None":
                        config["ckpt_name"] = ckpt_name
                elif model_type == "UNet Model":
                    if unet_name != "None":
                        config["unet_name"] = unet_name
                    # UNet-specific settings
                    config["weight_dtype"] = weight_dtype
                elif model_type == "Nunchaku Flux":
                    if nunchaku_name != "None":
                        config["nunchaku_name"] = nunchaku_name
                    # Nunchaku Flux-specific settings
                    config["data_type"] = data_type
                    config["cache_threshold"] = cache_threshold
                    config["attention"] = attention
                    config["i2f_mode"] = i2f_mode
                    config["cpu_offload"] = cpu_offload
                elif model_type == "Nunchaku Qwen":
                    if qwen_name != "None":
                        config["qwen_name"] = qwen_name
                    # Nunchaku Qwen-specific settings (only offload parameters are used)
                    config["cpu_offload"] = cpu_offload
                    config["num_blocks_on_gpu"] = num_blocks_on_gpu
                    config["use_pin_memory"] = use_pin_memory
                elif model_type == "GGUF Model":
                    if gguf_name != "None":
                        config["gguf_name"] = gguf_name
                    # GGUF-specific settings
                    config["gguf_dequant_dtype"] = gguf_dequant_dtype
                    config["gguf_patch_dtype"] = gguf_patch_dtype
                    config["gguf_patch_on_device"] = gguf_patch_on_device
                
                # Only save CLIP settings if configure_clip is enabled
                if configure_clip:
                    config["clip_source"] = clip_source
                    
                    # CLIP layer trimming only applies to Standard Checkpoints with baked CLIP
                    if model_type == "Standard Checkpoint":
                        config["enable_clip_layer"] = enable_clip_layer
                        config["stop_at_clip_layer"] = stop_at_clip_layer
                    
                    # CLIP configuration for External sources (UNet, Nunchaku, GGUF)
                    if clip_source == "External":
                        config["clip_count"] = clip_count
                        config["clip_type"] = clip_type
                        
                        # Only save CLIP names if not "None"
                        if clip_name1 != "None":
                            config["clip_name1"] = clip_name1
                        if clip_name2 != "None":
                            config["clip_name2"] = clip_name2
                        if clip_name3 != "None":
                            config["clip_name3"] = clip_name3
                        if clip_name4 != "None":
                            config["clip_name4"] = clip_name4
                
                # Only save VAE settings if configure_vae is enabled
                if configure_vae:
                    config["vae_source"] = vae_source
                    # Only save VAE name if using External source
                    if vae_source == "External" and vae_name != "None":
                        config["vae_name"] = vae_name
                
                # Only save latent settings if configure_latent is enabled
                if configure_latent:
                    config["resolution"] = resolution
                    # batch_size is intentionally not saved - it's a local/workflow value
                
                # Only save sampler settings if configure_sampler is enabled
                if configure_sampler:
                    config["sampler_name"] = sampler_name
                    config["scheduler"] = scheduler
                    config["steps"] = steps
                    config["cfg"] = cfg
                    # flux_guidance only relevant for Flux models (check clip_type for external CLIP models)
                    if model_type == "Nunchaku Flux" or (clip_type == "flux" and model_type in ["UNet Model", "GGUF Model"]):
                        config["flux_guidance"] = flux_guidance
                
                # Only save LoRA settings if configure_model_only_lora is enabled
                if configure_model_only_lora:
                    config["lora_count"] = lora_count
                    # Save all LoRA settings (even if switches are off)
                    for i in range(1, 4):
                        config[f"lora_switch_{i}"] = kwargs.get(f'lora_switch_{i}', False)
                        config[f"lora_name_{i}"] = kwargs.get(f'lora_name_{i}', 'None')
                        config[f"lora_weight_{i}"] = kwargs.get(f'lora_weight_{i}', 1.0)
                
                # Only save Model Sampling settings if configure_model_sampling is enabled (snapshot approach)
                if configure_model_sampling:
                    config["sampling_method"] = sampling_method
                    
                    # Always save shift (universal parameter)
                    config["shift"] = shift
                    
                    # Save method-specific parameters based on active sampling method
                    if sampling_method == "Flux" or sampling_method == "LTXV":
                        config["base_shift"] = base_shift
                    if sampling_method == "Flux":
                        config["sampling_width"] = sampling_width
                        config["sampling_height"] = sampling_height
                    elif sampling_method == "LCM":
                        config["original_timesteps"] = original_timesteps
                        config["zsnr"] = zsnr
                    elif sampling_method == "ContinuousEDM":
                        config["sampling_subtype"] = sampling_subtype
                        config["sigma_max"] = sigma_max
                        config["sigma_min"] = sigma_min
                    elif sampling_method == "ContinuousV":
                        config["sigma_max"] = sigma_max
                        config["sigma_min"] = sigma_min
                
                # Only add width/height if using Custom resolution
                if configure_latent and resolution == "Custom":
                    config["width"] = width
                    config["height"] = height
                
                if save_template(new_template_name.strip(), config):
                    cstr(f"✓ Template '{new_template_name}' saved successfully").msg.print()
                else:
                    cstr(f"✗ Failed to save template '{new_template_name}'").error.print()
            # Stop execution - template saved, no model loading needed
            empty_pipe = {"model": None, "clip": None, "vae": None, "latent": None}
            nodes.interrupt_processing()
            return (empty_pipe,)
        
        elif template_action == "Delete":
            if template_name and template_name != "None":
                if delete_template(template_name):
                    cstr(f"✓ Template '{template_name}' deleted successfully").msg.print()
                else:
                    cstr(f"✗ Failed to delete template '{template_name}'").error.print()
            # Stop execution - template deleted, no model loading needed
            empty_pipe = {"model": None, "clip": None, "vae": None, "latent": None}
            nodes.interrupt_processing()
            return (empty_pipe,)
        
        # Normalize inputs
        configure_clip = bool(configure_clip)
        configure_vae = bool(configure_vae)
        configure_latent = bool(configure_latent)
        configure_sampler = bool(configure_sampler)
        configure_model_only_lora = bool(configure_model_only_lora)
        configure_model_sampling = bool(configure_model_sampling)
        enable_clip_layer = bool(enable_clip_layer)
        clip_count_int = int(clip_count)
        lora_count_int = int(lora_count)
        
        is_standard = (model_type == "Standard Checkpoint")
        is_unet = (model_type == "UNet Model")
        is_nunchaku = (model_type == "Nunchaku Flux")
        is_qwen = (model_type == "Nunchaku Qwen")
        is_gguf = (model_type == "GGUF Model")
        use_baked_clip = (clip_source == "Baked")
        use_baked_vae = (vae_source == "Baked")
        
        loaded_model = None
        loaded_clip = None
        loaded_vae = None
        ckpt_parts = None
        checkpoint_name = ""
        
        safe_exts = {".safetensors", ".sft"}
        
        # ============================================================
        # STEP 0: Pre-Load Memory Cleanup
        # ============================================================
        
        if memory_cleanup:
            cleanup_memory_before_load()
        
        # ============================================================
        # STEP 1: Load Model (Standard Checkpoint, UNet, Nunchaku Flux, Nunchaku Qwen, or GGUF)
        # ============================================================
        
        if is_standard:
            # Load standard checkpoint
            if ckpt_name in (None, '', 'None'):
                raise ValueError("Please select a checkpoint file")
            
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if not ckpt_path or not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")
            
            _, ext = os.path.splitext(ckpt_path.lower())
            if ext not in safe_exts:
                cstr(f"Warning: '{ckpt_name}' uses extension '{ext}'. Consider .safetensors for safety.").warning.print()
            
            if not os.access(ckpt_path, os.R_OK):
                raise RuntimeError(f"Checkpoint file not readable: {ckpt_path}")
            
            # Load checkpoint with conditional outputs
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=use_baked_vae if configure_vae else False,
                output_clip=use_baked_clip if configure_clip else False,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            
            checkpoint_name = ckpt_name
            ckpt_parts = loaded_ckpt[:3] if hasattr(loaded_ckpt, '__len__') and len(loaded_ckpt) >= 3 else None
            loaded_model = ckpt_parts[0] if ckpt_parts else loaded_ckpt
            
        elif is_nunchaku:
            # ============================================================
            # STEP 1B: Load Nunchaku Quantized Model
            # ============================================================
            
            if nunchaku_name in (None, '', 'None'):
                raise ValueError("Please select a Nunchaku model file")
            
            nunchaku_path = folder_paths.get_full_path("diffusion_models", nunchaku_name)
            if not nunchaku_path or not os.path.isfile(nunchaku_path):
                raise FileNotFoundError(f"Nunchaku model not found: {nunchaku_name}")
            
            _, ext = os.path.splitext(nunchaku_path.lower())
            if ext not in safe_exts:
                cstr(f"Warning: '{nunchaku_name}' uses extension '{ext}'. Consider .safetensors.").warning.print()
            
            if not os.access(nunchaku_path, os.R_OK):
                raise RuntimeError(f"Nunchaku file not readable: {nunchaku_path}")
            
            if not NUNCHAKU_AVAILABLE:
                cstr("[Nunchaku Flux] ComfyUI-nunchaku extension not available - skipping model load").warning.print()
                cstr("[Nunchaku Flux] Install from: https://github.com/nunchaku-tech/ComfyUI-nunchaku").msg.print()
                loaded_model = None
                checkpoint_name = ""
            else:
                # Load with Nunchaku wrapper
                try:
                    cstr(f"[Nunchaku Flux] Loading quantized model: {nunchaku_name}").msg.print()
                    
                    loaded_model = load_nunchaku_model(
                        model_path=nunchaku_path,
                        device=None,  # Auto-detect
                        dtype=None,  # Will be determined from data_type
                        cpu_offload=(cpu_offload == "enable" or cpu_offload == "auto"),
                        cache_threshold=cache_threshold,
                        attention=attention,
                        data_type=data_type,
                        i2f_mode=i2f_mode,
                        model_type="flux"
                    )
                    
                    # Set checkpoint name from the model file
                    checkpoint_name = nunchaku_name
                    
                except Exception as e:
                    cstr(f"[Nunchaku Flux] Failed to load model '{nunchaku_name}': {e}").error.print()
                    loaded_model = None
                    checkpoint_name = ""
        
        elif is_qwen:
            # ============================================================
            # STEP 1D: Load Nunchaku Qwen Model
            # ============================================================
            
            if qwen_name in (None, '', 'None'):
                raise ValueError("Please select a Nunchaku Qwen model file")
            
            qwen_path = folder_paths.get_full_path("diffusion_models", qwen_name)
            if not qwen_path or not os.path.isfile(qwen_path):
                raise FileNotFoundError(f"Nunchaku Qwen model not found: {qwen_name}")
            
            _, ext = os.path.splitext(qwen_path.lower())
            if ext not in safe_exts:
                cstr(f"Warning: '{qwen_name}' uses extension '{ext}'. Consider .safetensors.").warning.print()
            
            if not os.access(qwen_path, os.R_OK):
                raise RuntimeError(f"Qwen file not readable: {qwen_path}")
            
            if not NUNCHAKU_AVAILABLE:
                cstr("[Nunchaku Qwen] ComfyUI-nunchaku extension not available - skipping model load").warning.print()
                cstr("[Nunchaku Qwen] Install from: https://github.com/nunchaku-tech/ComfyUI-nunchaku").msg.print()
                loaded_model = None
                checkpoint_name = ""
            else:
                # Load Nunchaku Qwen model
                checkpoint_name = qwen_name
                
                try:
                    loaded_model = load_nunchaku_model(
                        model_path=qwen_path,
                        device=None,  # Auto-detect
                        dtype=None,  # Auto-detect
                        cpu_offload=(cpu_offload == "enable" or cpu_offload == "auto"),
                        num_blocks_on_gpu=num_blocks_on_gpu,
                        use_pin_memory=(use_pin_memory == "enable"),
                        model_type="qwen"
                    )
                    
                except Exception as e:
                    cstr(f"[Nunchaku Qwen] Failed to load model '{qwen_name}': {e}").error.print()
                    loaded_model = None
                    checkpoint_name = ""
        
        elif is_gguf:
            # ============================================================
            # STEP 1E: Load GGUF Quantized Model
            # ============================================================
            
            if gguf_name in (None, '', 'None'):
                raise ValueError("Please select a GGUF model file")
            
            gguf_path = folder_paths.get_full_path("diffusion_models", gguf_name)
            if not gguf_path or not os.path.isfile(gguf_path):
                raise FileNotFoundError(f"GGUF model not found: {gguf_name}")
            
            if not gguf_path.lower().endswith('.gguf'):
                cstr(f"Warning: '{gguf_name}' doesn't have .gguf extension").warning.print()
            
            if not os.access(gguf_path, os.R_OK):
                raise RuntimeError(f"GGUF file not readable: {gguf_path}")
            
            if not GGUF_AVAILABLE:
                cstr("[GGUF] ComfyUI-GGUF extension not available - skipping model load").warning.print()
                cstr("[GGUF] Install from: https://github.com/city96/ComfyUI-GGUF").msg.print()
                loaded_model = None
                checkpoint_name = ""
            else:
                # Load GGUF model
                checkpoint_name = gguf_name
                
                try:
                    loaded_model = load_gguf_model(
                        model_path=gguf_path,
                        dequant_dtype=gguf_dequant_dtype,
                        patch_dtype=gguf_patch_dtype,
                        patch_on_device=gguf_patch_on_device
                    )
                    
                except Exception as e:
                    cstr(f"[GGUF] Failed to load model '{gguf_name}': {e}").error.print()
                    loaded_model = None
                    checkpoint_name = ""
            
        elif is_unet:
            # ============================================================
            # STEP 1F: Load Standard UNet Model
            # ============================================================
            
            if unet_name in (None, '', 'None'):
                raise ValueError("Please select a UNet model file")
            
            unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
            if not unet_path or not os.path.isfile(unet_path):
                raise FileNotFoundError(f"UNet model not found: {unet_name}")
            
            _, ext = os.path.splitext(unet_path.lower())
            if ext not in safe_exts:
                cstr(f"Warning: '{unet_name}' uses extension '{ext}'. Consider .safetensors.").warning.print()
            
            if not os.access(unet_path, os.R_OK):
                raise RuntimeError(f"UNet file not readable: {unet_path}")
            
            # Check if we need baked components (CLIP or VAE)
            needs_baked_clip = configure_clip and use_baked_clip
            needs_baked_vae = configure_vae and use_baked_vae
            needs_vae_for_latent = configure_latent and not configure_vae
            
            if needs_baked_clip or needs_baked_vae or needs_vae_for_latent:
                # Try to load as checkpoint to extract baked components
                try:
                    loaded_ckpt = comfy.sd.load_checkpoint_guess_config(
                        unet_path,
                        output_vae=(needs_baked_vae or needs_vae_for_latent),
                        output_clip=needs_baked_clip,
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    )
                    
                    ckpt_parts = loaded_ckpt[:3] if hasattr(loaded_ckpt, '__len__') and len(loaded_ckpt) >= 3 else None
                    loaded_model = ckpt_parts[0] if ckpt_parts else loaded_ckpt
                    checkpoint_name = unet_name
                    
                except Exception as e:
                    # If checkpoint loading fails, fall back to diffusion model loading
                    cstr(f"Note: UNet file doesn't contain baked components: {e}").msg.print()
                    
                    # Configure model options
                    model_options: dict[str, Any] = {}
                    if weight_dtype == "fp8_e4m3fn":
                        model_options["dtype"] = torch.float8_e4m3fn
                    elif weight_dtype == "fp8_e4m3fn_fast":
                        model_options["dtype"] = torch.float8_e4m3fn
                        model_options["fp8_optimizations"] = True
                    elif weight_dtype == "fp8_e5m2":
                        model_options["dtype"] = torch.float8_e5m2
                    
                    loaded_model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                    checkpoint_name = unet_name
            else:
                # No baked components needed - use standard diffusion model loading
                model_options = {}
                if weight_dtype == "fp8_e4m3fn":
                    model_options["dtype"] = torch.float8_e4m3fn
                elif weight_dtype == "fp8_e4m3fn_fast":
                    model_options["dtype"] = torch.float8_e4m3fn
                    model_options["fp8_optimizations"] = True
                elif weight_dtype == "fp8_e5m2":
                    model_options["dtype"] = torch.float8_e5m2
                
                loaded_model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                checkpoint_name = unet_name
        
        else:
            raise ValueError("Invalid model_type. Choose 'Standard Checkpoint', 'UNet Model', 'Nunchaku Flux', 'Nunchaku Qwen', or 'GGUF Model'")
        
        # ============================================================
        # STEP 2: Load CLIP (if configured)
        # ============================================================
        
        if configure_clip:
            if use_baked_clip:
                # Use baked CLIP from checkpoint (or UNet if it has one)
                # Note: Quantized models don't have baked CLIP
                if is_nunchaku or is_qwen or is_gguf:
                    if is_nunchaku:
                        model_label = "Nunchaku Flux"
                    elif is_qwen:
                        model_label = "Nunchaku Qwen"
                    else:
                        model_label = "GGUF"
                    cstr(f"[{model_label}] Quantized models don't contain baked CLIP - please use External CLIP").warning.print()
                elif ckpt_parts and ckpt_parts[1]:
                    base_clip = ckpt_parts[1]
                    if enable_clip_layer:
                        loaded_clip = base_clip.clone()
                        loaded_clip.clip_layer(stop_at_clip_layer)
                    else:
                        loaded_clip = base_clip
                else:
                    cstr("Warning: Baked CLIP requested but not found in checkpoint").warning.print()
            
            else:
                # Load external CLIP files
                clip_paths = []
                clip_names = [clip_name1, clip_name2, clip_name3, clip_name4]
                
                for i in range(clip_count_int):
                    clip_name = clip_names[i] if i < len(clip_names) else "None"
                    if clip_name not in (None, '', 'None'):
                        clip_path = folder_paths.get_full_path("clip", clip_name)
                        if clip_path and os.path.isfile(clip_path):
                            clip_paths.append(clip_path)
                        else:
                            cstr(f"Warning: CLIP file '{clip_name}' not found, skipping").warning.print()
                
                if not clip_paths:
                    raise ValueError("No valid CLIP files found. Please select at least one CLIP model")
                
                # Map clip_type string to CLIPType enum
                clip_type_map = {
                    "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
                    "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,
                    "sd3": comfy.sd.CLIPType.SD3,
                    "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,
                    "hunyuan_dit": comfy.sd.CLIPType.HUNYUAN_DIT,
                    "flux": comfy.sd.CLIPType.FLUX,
                    "mochi": comfy.sd.CLIPType.MOCHI,
                    "ltxv": comfy.sd.CLIPType.LTXV,
                    "hunyuan_video": comfy.sd.CLIPType.HUNYUAN_VIDEO,
                    "pixart": comfy.sd.CLIPType.PIXART,
                    "cosmos": comfy.sd.CLIPType.COSMOS,
                    "lumina2": comfy.sd.CLIPType.LUMINA2,
                    "wan": comfy.sd.CLIPType.WAN,
                    "hidream": comfy.sd.CLIPType.HIDREAM,
                    "chroma": comfy.sd.CLIPType.CHROMA,
                    "ace": comfy.sd.CLIPType.ACE,
                    "omnigen2": comfy.sd.CLIPType.OMNIGEN2,
                    "qwen_image": comfy.sd.CLIPType.QWEN_IMAGE,
                    "hunyuan_image": comfy.sd.CLIPType.HUNYUAN_IMAGE,
                }
                resolved_clip_type = clip_type_map.get(clip_type, comfy.sd.CLIPType.STABLE_DIFFUSION)
                
                loaded_clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=resolved_clip_type,
                )
        
        # ============================================================
        # STEP 3: Load VAE (if configured or needed for latent channel detection)
        # ============================================================
        
        # If latent is configured, we need VAE for channel detection even if configure_vae is False
        needs_vae_for_latent = configure_latent and not configure_vae
        
        if configure_vae or needs_vae_for_latent:
            if use_baked_vae or needs_vae_for_latent:
                # Use baked VAE from checkpoint (or UNet if it has one)
                # Note: Nunchaku models don't have baked VAE
                if is_nunchaku:
                    if needs_vae_for_latent:
                        cstr("[Nunchaku] Nunchaku models don't contain baked VAE - please enable 'Configure VAE' and use External VAE").warning.print()
                elif ckpt_parts and ckpt_parts[2]:
                    loaded_vae = ckpt_parts[2]
                else:
                    # Fastfail: If latent is needed but no baked VAE exists
                    if needs_vae_for_latent:
                        raise ValueError(
                            "Cannot create latent: Model has no baked VAE. "
                            "Please enable 'Configure VAE' and set vae_source to 'External', "
                            "or disable 'Configure Latent'."
                        )
                    cstr("Warning: Baked VAE requested but not found in model").warning.print()
            
            elif configure_vae and not use_baked_vae:
                # Load external VAE file
                if vae_name in (None, '', 'None'):
                    cstr("Warning: External VAE requested but none selected").warning.print()
                else:
                    vae_path = folder_paths.get_full_path("vae", vae_name)
                    if vae_path and os.path.isfile(vae_path):
                        vae_sd = comfy.utils.load_torch_file(vae_path)
                        loaded_vae = comfy.sd.VAE(sd=vae_sd)
                    else:
                        cstr(f"Warning: VAE file '{vae_name}' not found").warning.print()
        
        # ============================================================
        # STEP 4: Apply LoRAs (if configured)
        # ============================================================
        
        if configure_model_only_lora:
            # Collect enabled LoRAs
            lora_params = []
            for i in range(1, lora_count_int + 1):
                lora_switch = kwargs.get(f'lora_switch_{i}', False)
                lora_name = kwargs.get(f'lora_name_{i}', 'None')
                lora_weight = kwargs.get(f'lora_weight_{i}', 1.0)
                
                if lora_switch and lora_name not in (None, '', 'None'):
                    lora_params.append((lora_name, lora_weight))
            
            # Apply LoRAs if any enabled
            if lora_params:
                cstr(f"[LoRA] Applying {len(lora_params)} LoRA(s)...").msg.print()
                loaded_model, loaded_clip = apply_loras_to_model(loaded_model, loaded_clip, lora_params)
        
        # ============================================================
        # STEP 4.5: Apply Model Sampling (if configured)
        # ============================================================
        
        if configure_model_sampling and loaded_model is not None:
            # Auto-fill Flux dimensions from latent config if available
            flux_width = sampling_width
            flux_height = sampling_height
            
            if configure_latent and sampling_method == "Flux":
                # Map preset resolution to width/height
                if resolution != "Custom" and resolution in self.resolution_map:
                    auto_width, auto_height = self.resolution_map[resolution]
                    flux_width = auto_width
                    flux_height = auto_height
                else:
                    # Use custom dimensions
                    flux_width = width
                    flux_height = height
                
                cstr(f"[Model Sampling] Auto-filled Flux dimensions from latent: {flux_width}x{flux_height}").msg.print()
            
            loaded_model = apply_model_sampling(
                loaded_model, 
                sampling_method=sampling_method,
                shift=shift,
                base_shift=base_shift,
                width=flux_width,
                height=flux_height,
                original_timesteps=original_timesteps,
                zsnr=zsnr,
                sampling_subtype=sampling_subtype,
                sigma_max=sigma_max,
                sigma_min=sigma_min
            )
        
        # ============================================================
        # STEP 5: Create Latent Tensor (if configured)
        # ============================================================
        
        latent_tensor = None
        final_width = width
        final_height = height
        
        if configure_latent:
            # Map preset resolution to width/height
            if resolution != "Custom" and resolution in self.resolution_map:
                final_width, final_height = self.resolution_map[resolution]
            
            # Detect latent channels from VAE
            detected_channels = LATENT_CHANNELS
            if loaded_vae:
                detected_channels = _detect_latent_channels_from_vae_obj(loaded_vae)
            
            # Create latent tensor
            latent_tensor = torch.zeros([
                batch_size,
                detected_channels,
                final_height // UNET_DOWNSAMPLE,
                final_width // UNET_DOWNSAMPLE
            ], device="cpu")
        
        # ============================================================
        # STEP 6: Construct output pipe with sampler settings
        # ============================================================
        
        pipe = {
            "model": loaded_model,
            "clip": loaded_clip if configure_clip else None,
            "vae": loaded_vae if configure_vae else None,
            "latent": {"samples": latent_tensor} if latent_tensor is not None else None,
            "width": final_width if configure_latent else None,
            "height": final_height if configure_latent else None,
            "batch_size": batch_size if configure_latent else None,
            "model_name": checkpoint_name,
            "vae_name": vae_name if not use_baked_vae and vae_name not in (None, '', 'None') else '',
            "clip_skip": stop_at_clip_layer if is_standard and use_baked_clip and enable_clip_layer else None,
            "is_nunchaku": is_nunchaku,
            "flux_guidance": flux_guidance,
        }
        
        # Add sampler settings if configured
        if configure_sampler:
            pipe["sampler_name"] = sampler_name
            pipe["scheduler"] = scheduler
            pipe["steps"] = steps
            pipe["cfg"] = cfg
            pipe["_allow_overwrite"] = False
        
        return (pipe,)


NODE_NAME = 'Smart Loader Plus [Eclipse]'
NODE_DESC = 'Smart Loader Plus'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLoader_SmartLoader_Plus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
