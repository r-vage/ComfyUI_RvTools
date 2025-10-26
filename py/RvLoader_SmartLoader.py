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
Smart Loader - Simplified Model Loader with Multi-Format Support

Streamlined model loader supporting multiple model formats and quantization methods:
- Standard Checkpoints (.safetensors, .ckpt)
- UNet-only models
- Nunchaku quantized models (Flux and Qwen-Image with SVDQuant INT4/FP4/FP8)
- GGUF quantized models (INT4/INT8 quantization)

Features:
- Automatic model type detection
- Format-specific loading options (cache, attention, offload)
- Template system for saving/loading configurations
- Graceful fallback when extensions are not installed
- Comprehensive VRAM management and cleanup
- Simplified interface without latent/sampler configuration (use separate nodes)
"""

from typing import Any
import os
import comfy
import comfy.sd
import torch
import folder_paths
import comfy.utils
import json
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

# Register custom folder path for GGUF files (if GGUF is available)
if GGUF_AVAILABLE:
    # Register "diffusion_models_gguf" key that filters for .gguf files
    base = folder_paths.folder_names_and_paths.get("diffusion_models_gguf", ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    orig, _ = folder_paths.folder_names_and_paths.get("diffusion_models", ([], {}))
    folder_paths.folder_names_and_paths["diffusion_models_gguf"] = (orig or base, {".gguf"})

MAX_RESOLUTION = 32768
LATENT_CHANNELS = 4
UNET_DOWNSAMPLE = 8

# Template system - shared with SmartLoader
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "json", "loader_templates")

def cleanup_memory_before_load():
    """
    Clean up memory before loading a new model.
    
    This function performs comprehensive memory cleanup:
    1. Python garbage collection
    2. Clear CUDA cache on all GPUs
    3. Clear CUDA IPC shared memory cache
    4. ComfyUI's soft empty cache
    
    Based on comfyui-multigpu's soft_empty_cache_multigpu function.
    """
    import gc
    import comfy.model_management as mm
    
    cstr("[Memory Cleanup] Starting pre-load memory cleanup...").msg.print()
    
    # Step 1: Python garbage collection
    gc.collect()
    
    # Step 2: Clear CUDA cache on all devices
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        cstr(f"[Memory Cleanup] Clearing CUDA cache on {device_count} device(s)").msg.print()
        
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                # Clear IPC shared memory cache if available
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
    
    # Step 3: Clear MPS cache (Apple Silicon)
    if hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'):
        try:
            torch.mps.empty_cache()
            cstr("[Memory Cleanup] Cleared MPS cache").msg.print()
        except Exception:
            pass
    
    # Step 4: ComfyUI's soft empty cache
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
                templates.append(f[:-5])  # Remove .json extension
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
    """Infer latent channel count from a VAE-like object. Fall back to default."""
    try:
        if hasattr(vae_obj, 'channels') and isinstance(getattr(vae_obj, 'channels'), int):
            return getattr(vae_obj, 'channels')
        if hasattr(vae_obj, 'latent_channels') and isinstance(getattr(vae_obj, 'latent_channels'), int):
            return getattr(vae_obj, 'latent_channels')
        for attr in ('encoder', 'conv_in', 'down_blocks'):
            sub = getattr(vae_obj, attr, None)
            if sub is not None and hasattr(sub, 'weight'):
                w = getattr(sub, 'weight')
                try:
                    if hasattr(w, 'ndim') and w.ndim == 4:
                        return int(w.shape[0])
                except Exception:
                    pass
    except Exception:
        pass
    return LATENT_CHANNELS

# Module-level flag to print support messages only once
_support_messages_printed = False

class RvLoader_SmartLoader:
    resolution = RESOLUTION_PRESETS
    resolution_map = RESOLUTION_MAP
    
    def __init__(self):
        # Print support availability only once per session
        global _support_messages_printed
        if not _support_messages_printed:
            _support_messages_printed = True
            
            # Print Nunchaku availability
            nunchaku_info = get_nunchaku_info()
            if nunchaku_info['available']:
                cstr(f"[SmartLoader] Nunchaku support enabled").msg.print()
            
            # Print GGUF availability
            if GGUF_AVAILABLE:
                cstr("[SmartLoader] GGUF support enabled").msg.print()

    @classmethod
    def INPUT_TYPES(cls):
        # Get Nunchaku availability for dynamic options
        nunchaku_info = get_nunchaku_info()
        
        # Base weight dtype options
        weight_dtype_options = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
        
        # Nunchaku-specific options (only show if available)
        nunchaku_cache_visible = nunchaku_info['available']
        nunchaku_attention_visible = nunchaku_info['available']
        nunchaku_offload_visible = nunchaku_info['available']
        
        inputs = {
            "required": {
                # === STEP 0: Template Management ===
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
                
                # === STEP 1: Model Type Selection ===
                "model_type": (["Standard Checkpoint", "UNet Model", "Nunchaku Flux", "Nunchaku Qwen", "GGUF Model"], {
                    "default": "Standard Checkpoint",
                    "tooltip": "Select model type: Standard (checkpoint), UNet (standalone diffusion model), Nunchaku Flux/Qwen (quantized), or GGUF (quantized)"
                }),
                
                # === STEP 2: Model Selection ===
                "ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), {
                    "default": "None",
                    "tooltip": "Select checkpoint file (for Standard Checkpoint mode)"
                }),
                "unet_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select UNet diffusion model (for UNet Model mode)"
                }),
                "nunchaku_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select Nunchaku quantized Flux model (for Nunchaku Flux mode)"
                }),
                "qwen_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "default": "None",
                    "tooltip": "Select Nunchaku quantized Qwen model (for Nunchaku Qwen mode)"
                }),
                "gguf_name": (["None"] + folder_paths.get_filename_list("diffusion_models_gguf"), {
                    "default": "None",
                    "tooltip": "Select GGUF quantized model (.gguf file)"
                }),
                "weight_dtype": (weight_dtype_options, {
                    "default": "default",
                    "tooltip": "Weight dtype for UNet model (applies fp8 quantization)"
                }),
                
                # === STEP 2b: Nunchaku Flux Options ===
                "data_type": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                    "tooltip": "Model data type for Nunchaku Flux. Use bfloat16 for 30/40-series GPUs, float16 for 20-series GPUs."
                }),
                "cache_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "First-block caching for Nunchaku Flux (0=disabled, 0.12=typical). Higher=faster but lower quality."
                }),
                "attention": (["flash-attention2", "nunchaku-fp16"], {
                    "default": "flash-attention2",
                    "tooltip": "Attention implementation for Nunchaku Flux. Use nunchaku-fp16 for 20-series GPUs."
                }),
                "i2f_mode": (["enabled", "always"], {
                    "default": "enabled",
                    "tooltip": "GEMM implementation for 20-series GPUs (ignored on other GPUs)."
                }),
                
                # === STEP 2c: Shared Nunchaku Options ===
                "cpu_offload": (["auto", "enable", "disable"], {
                    "default": "auto",
                    "tooltip": "CPU offload for Nunchaku models. Auto enables for <14GB VRAM."
                }),
                
                # === STEP 2d: Nunchaku Qwen Options ===
                "num_blocks_on_gpu": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Number of transformer blocks to keep on GPU (for Nunchaku Qwen)"
                }),
                "use_pin_memory": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Use pinned memory for faster CPU-GPU transfers (for Nunchaku Qwen)"
                }),
                
                # === STEP 2e: GGUF Options ===
                "gguf_dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "Dequantization dtype for GGUF (default=auto, target=match model)"
                }),
                "gguf_patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "LoRA patch dtype for GGUF"
                }),
                "gguf_patch_on_device": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply LoRA patches on GPU for GGUF (faster but uses more VRAM)"
                }),
                
                # === STEP 3: Component Configuration Toggles ===
                "configure_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable CLIP configuration options"
                }),
                "configure_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE configuration options"
                }),
                
                # === STEP 4a: CLIP Configuration ===
                "clip_source": (["Baked", "External"], {
                    "default": "Baked",
                    "tooltip": "Baked = use checkpoint's CLIP | External = load separate CLIP files"
                }),
                "clip_count": (["1", "2", "3", "4"], {
                    "default": "1",
                    "tooltip": "Number of CLIP models to load (for ensemble)"
                }),
                "clip_name1": (["None"] + folder_paths.get_filename_list("clip"), {
                    "default": "None",
                    "tooltip": "Primary CLIP model"
                }),
                "clip_name2": (["None"] + folder_paths.get_filename_list("clip"), {
                    "default": "None",
                    "tooltip": "Secondary CLIP model (optional)"
                }),
                "clip_name3": (["None"] + folder_paths.get_filename_list("clip"), {
                    "default": "None",
                    "tooltip": "Third CLIP model (optional)"
                }),
                "clip_name4": (["None"] + folder_paths.get_filename_list("clip"), {
                    "default": "None",
                    "tooltip": "Fourth CLIP model (optional)"
                }),
                "clip_type": (["flux", "sd3", "sdxl", "stable_cascade", "stable_audio", "hunyuan_dit", "mochi", "ltxv", "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], {
                    "default": "flux",
                    "tooltip": "CLIP architecture type (for external CLIP)"
                }),
                "enable_clip_layer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trim CLIP to specific layer (saves memory)"
                }),
                "stop_at_clip_layer": ("INT", {
                    "default": -2,
                    "min": -24,
                    "max": -1,
                    "step": 1,
                    "tooltip": "CLIP layer to stop at (negative index, -1 = last layer)"
                }),
                
                # === STEP 4b: VAE Configuration ===
                "vae_source": (["Baked", "External"], {
                    "default": "Baked",
                    "tooltip": "Baked = use checkpoint's VAE | External = load separate VAE file"
                }),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"), {
                    "default": "None",
                    "tooltip": "External VAE file to load"
                }),
            },
        }
        
        return inputs

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LOADER.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force ComfyUI to refresh INPUT_TYPES to get updated template list"""
        import time
        # Return timestamp of template directory to detect changes
        template_dir = TEMPLATE_DIR
        if os.path.exists(template_dir):
            # Get the latest modification time of any file in the directory
            try:
                mtime = max(os.path.getmtime(os.path.join(template_dir, f)) 
                           for f in os.listdir(template_dir) if f.endswith('.json'))
                return str(mtime)
            except (ValueError, OSError):
                pass
        return str(time.time())

    def execute(
        self,
        template_action: str,
        template_name: str,
        new_template_name: str,
        model_type: str,
        ckpt_name: str,
        unet_name: str,
        nunchaku_name: str,
        qwen_name: str,
        gguf_name: str,
        weight_dtype: str,
        data_type: str,
        cache_threshold: float,
        attention: str,
        i2f_mode: str,
        cpu_offload: str,
        num_blocks_on_gpu: int,
        use_pin_memory: str,
        gguf_dequant_dtype: str,
        gguf_patch_dtype: str,
        gguf_patch_on_device: bool,
        configure_clip: bool,
        configure_vae: bool,
        clip_source: str,
        clip_count: str,
        clip_name1: str,
        clip_name2: str,
        clip_name3: str,
        clip_name4: str,
        clip_type: str,
        enable_clip_layer: bool,
        stop_at_clip_layer: int,
        vae_source: str,
        vae_name: str,
    ) -> tuple:
        
        # Handle template actions - Save and Delete interrupt, Load doesn't
        if template_action == "Save":
            if new_template_name and new_template_name.strip():
                config = {
                    "model_type": model_type,
                    "ckpt_name": ckpt_name,
                    "unet_name": unet_name,
                    "nunchaku_name": nunchaku_name,
                    "qwen_name": qwen_name,
                    "gguf_name": gguf_name,
                    "weight_dtype": weight_dtype,
                    "data_type": data_type,
                    "cache_threshold": cache_threshold,
                    "attention": attention,
                    "i2f_mode": i2f_mode,
                    "cpu_offload": cpu_offload,
                    "num_blocks_on_gpu": num_blocks_on_gpu,
                    "use_pin_memory": use_pin_memory,
                    "gguf_dequant_dtype": gguf_dequant_dtype,
                    "gguf_patch_dtype": gguf_patch_dtype,
                    "gguf_patch_on_device": gguf_patch_on_device,
                    "configure_clip": configure_clip,
                    "configure_vae": configure_vae,
                    "configure_latent": False,  # Simplified loader doesn't have latent config
                    "configure_sampler": False,  # Simplified loader doesn't have sampler config
                    "clip_source": clip_source,
                    "clip_count": clip_count,
                    "clip_name1": clip_name1,
                    "clip_name2": clip_name2,
                    "clip_name3": clip_name3,
                    "clip_name4": clip_name4,
                    "clip_type": clip_type,
                    "enable_clip_layer": enable_clip_layer,
                    "stop_at_clip_layer": stop_at_clip_layer,
                    "vae_source": vae_source,
                    "vae_name": vae_name,
                }
                if save_template(new_template_name.strip(), config):
                    cstr(f"✓ Template '{new_template_name}' saved successfully").msg.print()
                else:
                    cstr(f"✗ Failed to save template '{new_template_name}'").error.print()
            # Stop execution - template saved, no model loading needed
            empty_pipe = {"model": None, "clip": None, "vae": None, "latent": None, "ckpt_name": ""}
            nodes.interrupt_processing()
            return (empty_pipe,)
        
        elif template_action == "Delete":
            if template_name and template_name != "None":
                if delete_template(template_name):
                    cstr(f"✓ Template '{template_name}' deleted successfully").msg.print()
                else:
                    cstr(f"✗ Failed to delete template '{template_name}'").error.print()
            # Stop execution - template deleted, no model loading needed
            empty_pipe = {"model": None, "clip": None, "vae": None, "latent": None, "ckpt_name": ""}
            nodes.interrupt_processing()
            return (empty_pipe,)
        
        # Normalize inputs
        configure_clip = bool(configure_clip)
        configure_vae = bool(configure_vae)
        enable_clip_layer = bool(enable_clip_layer)
        clip_count_int = int(clip_count)
        
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
                raise RuntimeError(
                    f"Nunchaku Flux type selected but ComfyUI-nunchaku extension not available.\n\n"
                    f"To use Nunchaku quantized Flux models, install ComfyUI-nunchaku:\n"
                    f"  1. Navigate to: ComfyUI/custom_nodes/\n"
                    f"  2. Clone: git clone https://github.com/mit-han-lab/ComfyUI-nunchaku\n"
                    f"  3. Install: cd ComfyUI-nunchaku && pip install -r requirements.txt\n"
                    f"  4. Restart ComfyUI\n\n"
                    f"Or select 'Standard Checkpoint' or 'UNet Model' type instead."
                )
            
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
                raise RuntimeError(
                    f"Failed to load Nunchaku model '{nunchaku_name}':\n{e}\n\n"
                    f"This might indicate:\n"
                    f"  - Corrupted model file\n"
                    f"  - Incompatible Nunchaku version\n"
                    f"  - Insufficient VRAM (try enabling CPU offload)\n"
                )
        
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
                raise RuntimeError(
                    f"Failed to load Nunchaku Qwen model '{qwen_name}':\n{e}\n\n"
                    f"This might indicate:\n"
                    f"  - Corrupted model file\n"
                    f"  - Incompatible Nunchaku version\n"
                    f"  - Insufficient VRAM (try adjusting num_blocks_on_gpu)\n"
                )
        
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
                raise RuntimeError(
                    f"Failed to load GGUF model '{gguf_name}':\n{e}\n\n"
                    f"This might indicate:\n"
                    f"  - Corrupted model file\n"
                    f"  - Incompatible GGUF version\n"
                    f"  - ComfyUI-GGUF not installed\n"
                    f"  - Missing gguf package (pip install --upgrade gguf)\n"
                )
            
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
        # STEP 3: Load VAE (if configured)
        # ============================================================
        
        if configure_vae:
            if use_baked_vae:
                # Use baked VAE from checkpoint (or UNet if it has one)
                # Note: Nunchaku models don't have baked VAE
                if is_nunchaku:
                    cstr("[Nunchaku] Nunchaku models don't contain baked VAE - please enable 'Configure VAE' and use External VAE").warning.print()
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
                        vae_sd = comfy.utils.load_torch_file(vae_path)
                        loaded_vae = comfy.sd.VAE(sd=vae_sd)
                    else:
                        cstr(f"Warning: VAE file '{vae_name}' not found").warning.print()
        
        # ============================================================
        # STEP 4: Construct output pipe (simplified - no latent/sampler)
        # ============================================================
        
        pipe = {
            "model": loaded_model,
            "clip": loaded_clip if configure_clip else None,
            "vae": loaded_vae if configure_vae else None,
            "model_name": checkpoint_name,
            "vae_name": vae_name if not use_baked_vae and vae_name not in (None, '', 'None') else '',
            "clip_skip": stop_at_clip_layer if configure_clip and enable_clip_layer else None,
            "is_nunchaku": is_nunchaku,  # Track if Nunchaku model
        }
        
        return (pipe,)


NODE_NAME = 'Smart Loader [RvTools]'
NODE_DESC = 'Smart Loader'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvLoader_SmartLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
