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

# Smart Loader - Streamlined Model Loader with Integrated LoRA Support
#
# Streamlined model loader supporting multiple model formats and quantization methods:
# - Standard Checkpoints (.safetensors, .ckpt)
# - UNet-only models
# - GGUF quantized models (INT4/INT8 quantization)
#
# Features:
# - Automatic model type detection
# - Format-specific loading options (cache, attention, offload)
# - Template system for saving/loading configurations with intelligent field filtering
# - Model-only LoRA support with up to 3 slots
# - Graceful fallback when extensions are not installed
# - Comprehensive VRAM management and cleanup
# - Auto-fill template names for easy updates
# - No latent or sampler configuration (use separate nodes for those)

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

# Import GGUF wrapper
from ..core.gguf_wrapper import (
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
        clip_paths, clip_exts = folder_paths.folder_names_and_paths["clip"]
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
        te_paths, te_exts = folder_paths.folder_names_and_paths["text_encoders"]
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

def cleanup_memory_before_load():
    # Clean up memory before loading a new model.
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

def _detect_latent_channels_from_vae_obj(vae_obj) -> int:
    # Infer latent channel count from a VAE-like object.
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

def apply_loras_to_model(model: Any, clip: Any, lora_params: list) -> tuple:
    # Apply LoRAs to model using ComfyUI's loader.
    #
    # Parameters:
    #     model: The model to apply LoRAs to
    #     clip: The CLIP model
    #     lora_params: List of tuples (lora_name, model_weight)
    #
    # Returns:
    #     (modified_model, modified_clip)
    if not lora_params:
        return (model, clip)
    
    cstr("[LoRA] Applying LoRAs to model").msg.print()
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

_support_messages_printed = False

class RvLoader_SmartLoader_Basic:
    resolution = RESOLUTION_PRESETS
    resolution_map = RESOLUTION_MAP
    
    def __init__(self):
        global _support_messages_printed
        if not _support_messages_printed:
            _support_messages_printed = True
            
            if GGUF_AVAILABLE:
                cstr("✓ GGUF support available").msg.print()

    @classmethod
    def INPUT_TYPES(cls):
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
        
        return {
            "required": {
                "model_type": (["Standard Checkpoint", "UNet Model", "GGUF Model"], {"default": "Standard Checkpoint", "tooltip": "Select model type"}),
                "ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), {"default": "None", "tooltip": "Select checkpoint file"}),
                "unet_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "Select UNet diffusion model"}),
                "gguf_name": (["None"] + (folder_paths.get_filename_list("diffusion_models_gguf") if "diffusion_models_gguf" in folder_paths.folder_names_and_paths else []), {"default": "None", "tooltip": "Select GGUF model"}),
                "weight_dtype": (weight_dtype_options, {"default": "default", "tooltip": "Weight dtype for UNet model"}),
                "model_device": (["auto", "cpu"], {"default": "auto", "tooltip": "Device to load model on (auto=GPU if available, cpu=force CPU)"}),
                "gguf_dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default", "tooltip": "Dequantization dtype"}),
                "gguf_patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default", "tooltip": "LoRA patch dtype"}),
                "gguf_patch_on_device": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Apply patches on GPU"}),
                "configure_clip": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Enable CLIP configuration"}),
                "configure_vae": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Enable VAE configuration"}),
                "configure_model_only_lora": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Enable model-only LoRA configuration"}),
                "clip_source": (["Baked", "External"], {"default": "Baked", "tooltip": "CLIP source"}),
                "clip_count": (["1", "2", "3", "4"], {"default": "1", "tooltip": "Number of CLIP models"}),
                "clip_name1": (clips, {"default": "None", "tooltip": "Primary CLIP model"}),
                "clip_name2": (clips, {"default": "None", "tooltip": "Secondary CLIP model"}),
                "clip_name3": (clips, {"default": "None", "tooltip": "Third CLIP model"}),
                "clip_name4": (clips, {"default": "None", "tooltip": "Fourth CLIP model"}),
                "clip_type": (["flux", "sd3", "sdxl", "stable_cascade", "stable_audio", "hunyuan_dit", "mochi", "ltxv", "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], {"default": "flux", "tooltip": "CLIP architecture type"}),
                "clip_device": (["auto", "cpu"], {"default": "auto", "tooltip": "Device to load CLIP on (auto=GPU if available, cpu=force CPU)"}),
                "enable_clip_layer": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Trim CLIP to specific layer"}),
                "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1, "tooltip": "CLIP layer to stop at"}),
                "vae_source": (["Baked", "External"], {"default": "Baked", "tooltip": "VAE source"}),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"), {"default": "None", "tooltip": "External VAE file"}),
                "vae_device": (["auto", "cpu"], {"default": "auto", "tooltip": "Device to load VAE on (auto=GPU if available, cpu=force CPU)"}),
                "lora_count": (["1", "2", "3"], {"default": "1", "tooltip": "Number of LoRA slots to configure"}),
                "lora_switch_1": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA 1"}),
                "lora_name_1": (loras, {"default": "None", "tooltip": "LoRA 1 file"}),
                "lora_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 1 model weight"}),
                "lora_switch_2": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA 2"}),
                "lora_name_2": (loras, {"default": "None", "tooltip": "LoRA 2 file"}),
                "lora_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 2 model weight"}),
                "lora_switch_3": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA 3"}),
                "lora_name_3": (loras, {"default": "None", "tooltip": "LoRA 3 file"}),
                "lora_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA 3 model weight"}),
                "memory_cleanup": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Perform memory cleanup before loading"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LOADER.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        # Extract all parameters
        model_type = kwargs.get('model_type', 'Standard Checkpoint')
        ckpt_name = kwargs.get('ckpt_name', 'None')
        unet_name = kwargs.get('unet_name', 'None')
        gguf_name = kwargs.get('gguf_name', 'None')
        weight_dtype = kwargs.get('weight_dtype', 'default')
        model_device = kwargs.get('model_device', 'auto')
        
        gguf_dequant_dtype = kwargs.get('gguf_dequant_dtype', 'default')
        gguf_patch_dtype = kwargs.get('gguf_patch_dtype', 'default')
        gguf_patch_on_device = kwargs.get('gguf_patch_on_device', False)
        
        configure_clip = kwargs.get('configure_clip', True)
        configure_vae = kwargs.get('configure_vae', True)
        configure_model_only_lora = kwargs.get('configure_model_only_lora', False)
        
        clip_source = kwargs.get('clip_source', 'Baked')
        clip_count = kwargs.get('clip_count', '1')
        clip_name1 = kwargs.get('clip_name1', 'None')
        clip_name2 = kwargs.get('clip_name2', 'None')
        clip_name3 = kwargs.get('clip_name3', 'None')
        clip_name4 = kwargs.get('clip_name4', 'None')
        clip_type = kwargs.get('clip_type', 'flux')
        clip_device = kwargs.get('clip_device', 'auto')
        enable_clip_layer = kwargs.get('enable_clip_layer', True)
        stop_at_clip_layer = kwargs.get('stop_at_clip_layer', -2)
        
        vae_source = kwargs.get('vae_source', 'Baked')
        vae_name = kwargs.get('vae_name', 'None')
        vae_device = kwargs.get('vae_device', 'auto')
        
        lora_count = kwargs.get('lora_count', '1')
        
        memory_cleanup = kwargs.get('memory_cleanup', True)
        
        # Normalize inputs
        configure_clip = bool(configure_clip)
        configure_vae = bool(configure_vae)
        configure_model_only_lora = bool(configure_model_only_lora)
        enable_clip_layer = bool(enable_clip_layer)
        clip_count_int = int(clip_count)
        lora_count_int = int(lora_count)
        
        is_standard = (model_type == "Standard Checkpoint")
        is_unet = (model_type == "UNet Model")
        is_gguf = (model_type == "GGUF Model")
        use_baked_clip = (clip_source == "Baked")
        use_baked_vae = (vae_source == "Baked")
        
        # Resolve device settings
        if model_device == "auto":
            resolved_model_device = mm.get_torch_device()
        else:
            resolved_model_device = torch.device(model_device)
        
        if clip_device == "auto":
            resolved_clip_device = mm.text_encoder_device()
        else:
            resolved_clip_device = torch.device(clip_device)
        
        if vae_device == "auto":
            resolved_vae_device = mm.vae_device()
        else:
            resolved_vae_device = torch.device(vae_device)
        
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
        # STEP 1: Load Model (Standard Checkpoint, UNet, or GGUF)
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
            # Temporarily override device settings if forcing CPU
            original_unet_device = mm.unet_offload_device
            original_text_device = mm.text_encoder_device
            original_vae_device = mm.vae_device
            
            if model_device == "cpu":
                mm.unet_offload_device = lambda: torch.device("cpu")
            if clip_device == "cpu" and (configure_clip and use_baked_clip):
                mm.text_encoder_device = lambda: torch.device("cpu")
            if vae_device == "cpu" and (configure_vae and use_baked_vae):
                mm.vae_device = lambda: torch.device("cpu")
            
            try:
                loaded_ckpt = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=use_baked_vae if configure_vae else False,
                    output_clip=use_baked_clip if configure_clip else False,
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
            finally:
                # Restore original device functions
                mm.unet_offload_device = original_unet_device
                mm.text_encoder_device = original_text_device
                mm.vae_device = original_vae_device
            
            # Extract checkpoint parts
            checkpoint_name = ckpt_name
            ckpt_parts = loaded_ckpt[:3] if hasattr(loaded_ckpt, '__len__') and len(loaded_ckpt) >= 3 else None
            loaded_model = ckpt_parts[0] if ckpt_parts else loaded_ckpt
            
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
                
                cstr(f"[GGUF] Loading on device: {resolved_model_device}").msg.print()
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
            
            if needs_baked_clip or needs_baked_vae:
                # Try to load as checkpoint to extract baked components
                try:
                    loaded_ckpt = comfy.sd.load_checkpoint_guess_config(
                        unet_path,
                        output_vae=needs_baked_vae,
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
                    
                    cstr(f"[UNet] Target device: {resolved_model_device}").msg.print()
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
            raise ValueError("Invalid model_type. Choose 'Standard Checkpoint', 'UNet Model', or 'GGUF Model'")
        
        # ============================================================
        # STEP 2: Load CLIP (if configured)
        # ============================================================
        
        if configure_clip:
            if use_baked_clip:
                # Use baked CLIP from checkpoint (or UNet if it has one)
                # Note: GGUF models don't have baked CLIP
                if is_gguf:
                    cstr("[GGUF] Quantized models don't contain baked CLIP - please use External CLIP").warning.print()
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
                
                # Temporarily override device function if forcing CPU
                original_text_device = mm.text_encoder_device
                if clip_device == "cpu":
                    mm.text_encoder_device = lambda: torch.device("cpu")
                
                try:
                    loaded_clip = comfy.sd.load_clip(
                        ckpt_paths=clip_paths,
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                        clip_type=resolved_clip_type
                    )
                finally:
                    mm.text_encoder_device = original_text_device
                
        
        # ============================================================
        # STEP 3: Load VAE (if configured)
        # ============================================================
        
        if configure_vae:
            if use_baked_vae:
                # Use baked VAE from checkpoint (or UNet if it has one)
                # Note: GGUF models don't have baked VAE
                if is_gguf:
                    cstr("[GGUF] Quantized models don't contain baked VAE - please use External VAE").warning.print()
                elif ckpt_parts and ckpt_parts[2]:
                    loaded_vae = ckpt_parts[2]
                else:
                    cstr("Warning: Baked VAE requested but not found in model").warning.print()
            
            else:
                # Load external VAE file
                if vae_name in (None, '', 'None'):
                    cstr("Warning: External VAE requested but none selected").warning.print()
                else:
                    vae_path = folder_paths.get_full_path("vae", vae_name)
                    if vae_path and os.path.isfile(vae_path):
                        # Temporarily override device function if forcing CPU
                        original_vae_device = mm.vae_device
                        if vae_device == "cpu":
                            mm.vae_device = lambda: torch.device("cpu")
                        
                        try:
                            vae_sd = comfy.utils.load_torch_file(vae_path)
                            loaded_vae = comfy.sd.VAE(sd=vae_sd)
                        finally:
                            mm.vae_device = original_vae_device
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
        # STEP 5: Construct output pipe (no latent or sampler)
        # ============================================================
        
        pipe = {
            "model": loaded_model,
            "clip": loaded_clip if configure_clip else None,
            "vae": loaded_vae if configure_vae else None,
            "model_name": checkpoint_name,
            "vae_name": vae_name if not use_baked_vae and vae_name not in (None, '', 'None') else '',
            "clip_skip": stop_at_clip_layer if is_standard and use_baked_clip and enable_clip_layer else None,
        }
        
        return (pipe,)


NODE_NAME = 'Smart Loader Basic [Eclipse]'
NODE_DESC = 'Smart Loader Basic'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLoader_SmartLoader_Basic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
