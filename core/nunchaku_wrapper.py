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

# Nunchaku Model Wrapper for Smart Loader Plus
#
# This module provides detection and loading support for Nunchaku quantized models.
# Nunchaku models are quantized FLUX and Qwen models (INT4/FP4/FP8) that require special loading.
#
# Key Features:
# - Automatic detection via filename patterns and metadata inspection
# - Graceful fallback when ComfyUI-nunchaku is not installed
# - Hybrid detection: filename patterns + metadata checks
# - Compatible with ComfyUI ModelPatcher interface
# - LoRA support for both Flux and Qwen models

import os
import json
import re
import copy
from pathlib import Path
from collections import defaultdict
from typing import Optional, Any, TYPE_CHECKING, Dict, List, Tuple, Union, Callable
import torch
from torch import nn

# Import cstr for Eclipse-style logging
from . import cstr

#cstr("[Nunchaku Wrapper] Module loading started...").msg.print()

# Try to import Nunchaku - graceful fallback if not available
NUNCHAKU_AVAILABLE = False
NunchakuFluxTransformer2dModel: Optional[Any] = None
NunchakuQwenImageTransformer2DModel: Optional[Any] = None
apply_cache_on_transformer: Optional[Any] = None
ComfyFluxWrapper: Optional[Any] = None
QwenConfig: Optional[Any] = None
QwenModelBase: Optional[Any] = None
NunchakuModelPatcher: Optional[Any] = None

#cstr("[Nunchaku Wrapper] Starting Nunchaku imports...").msg.print()

try:
    from nunchaku import NunchakuFluxTransformer2dModel as _NunchakuFluxTransformer2dModel
    from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer as _apply_cache_on_transformer
    
    NunchakuFluxTransformer2dModel = _NunchakuFluxTransformer2dModel
    apply_cache_on_transformer = _apply_cache_on_transformer
    
    cstr("[Nunchaku Wrapper] ✓ Nunchaku base imports successful").msg.print()
    
    # Try to import Qwen model
    try:
        from nunchaku.models.qwenimage import NunchakuQwenImageTransformer2DModel as _NunchakuQwenImageTransformer2DModel
        NunchakuQwenImageTransformer2DModel = _NunchakuQwenImageTransformer2DModel
        #cstr("[Nunchaku Wrapper] ✓ Qwen model import successful").msg.print()
    except ImportError as e:
        #cstr(f"[Nunchaku Wrapper] Qwen model not available: {e}").warning.print()
        NunchakuQwenImageTransformer2DModel = None
    
    # Import ComfyFluxWrapper and Qwen classes from ComfyUI-nunchaku extension
    try:
        import sys
        from pathlib import Path
        
        # Look for ComfyUI-nunchaku in custom_nodes
        # Path: core/nunchaku_wrapper.py -> ComfyUI_Eclipse -> custom_nodes
        custom_nodes_path = Path(__file__).parent.parent.parent
        nunchaku_path = custom_nodes_path / "ComfyUI-nunchaku"
        
        #cstr(f"[Nunchaku Wrapper] Looking for ComfyUI-nunchaku at: {nunchaku_path}").msg.print()
        #cstr(f"[Nunchaku Wrapper] Path exists: {nunchaku_path.exists()}").msg.print()
        
        if nunchaku_path.exists():
            # Import classes using the EXACT same module path that PuLID uses
            # PuLID's relative imports create entries like 'D:\...\ComfyUI-nunchaku.wrappers.flux' in sys.modules
            # We need to check sys.modules to find the actual module name and reuse it
            # cstr("[Nunchaku Wrapper] Attempting to import Nunchaku classes...").msg.print()
            
            import importlib
            
            # Strategy: Check if PuLID has already imported the wrapper, and reuse that module
            wrapper_module = None
            for mod_name in sys.modules:
                if 'ComfyUI-nunchaku' in mod_name and 'wrappers.flux' in mod_name:
                    wrapper_module = sys.modules[mod_name]
                    #print(f"[Nunchaku Wrapper] Found existing wrapper module: {mod_name}")
                    break
            
            if wrapper_module is not None:
                # Reuse the already-imported module
                ComfyFluxWrapper = wrapper_module.ComfyFluxWrapper
                
                # Find and import other modules using similar pattern
                base_mod_name = mod_name.rsplit('.wrappers.flux', 1)[0]
                
                qwen_config_module = sys.modules.get(f"{base_mod_name}.model_configs.qwenimage")
                if qwen_config_module:
                    QwenConfig = qwen_config_module.NunchakuQwenImage
                else:
                    qwen_config_module = importlib.import_module(f"{base_mod_name}.model_configs.qwenimage")
                    QwenConfig = qwen_config_module.NunchakuQwenImage
                
                qwen_base_module = sys.modules.get(f"{base_mod_name}.model_base.qwenimage")
                if qwen_base_module:
                    QwenModelBase = qwen_base_module.NunchakuQwenImage
                else:
                    qwen_base_module = importlib.import_module(f"{base_mod_name}.model_base.qwenimage")
                    QwenModelBase = qwen_base_module.NunchakuQwenImage
                
                patcher_module = sys.modules.get(f"{base_mod_name}.model_patcher")
                if patcher_module:
                    NunchakuModelPatcher = patcher_module.NunchakuModelPatcher
                else:
                    patcher_module = importlib.import_module(f"{base_mod_name}.model_patcher")
                    NunchakuModelPatcher = patcher_module.NunchakuModelPatcher
                
               # print(f"[Nunchaku Wrapper] ✓ Reused existing modules with base: {base_mod_name}")
            else:
                # First import - use standard package import (ComfyUI will have set this up)
                # Try direct import first (ComfyUI __init__ system should have loaded it)
                try:
                    import ComfyUI_nunchaku
                    from ComfyUI_nunchaku.wrappers.flux import ComfyFluxWrapper as _ComfyFluxWrapper
                    ComfyFluxWrapper = _ComfyFluxWrapper
                    
                    from ComfyUI_nunchaku.model_configs.qwenimage import NunchakuQwenImage as _QwenConfig
                    QwenConfig = _QwenConfig
                    
                    from ComfyUI_nunchaku.model_base.qwenimage import NunchakuQwenImage as _QwenModelBase
                    QwenModelBase = _QwenModelBase
                    
                    from ComfyUI_nunchaku.model_patcher import NunchakuModelPatcher as _NunchakuModelPatcher
                    NunchakuModelPatcher = _NunchakuModelPatcher
                    
                  #  print(f"[Nunchaku Wrapper] ✓ Imported via ComfyUI_nunchaku package")
                except ImportError:
                    # Last resort: manual sys.path approach
                    nunchaku_parent = str(nunchaku_path.parent)
                    if nunchaku_parent not in sys.path:
                        sys.path.insert(0, nunchaku_parent)
                    
                    flux_wrapper_module = importlib.import_module("ComfyUI-nunchaku.wrappers.flux")
                    ComfyFluxWrapper = flux_wrapper_module.ComfyFluxWrapper
                    
                    qwen_config_module = importlib.import_module("ComfyUI-nunchaku.model_configs.qwenimage")
                    QwenConfig = qwen_config_module.NunchakuQwenImage
                    
                    qwen_base_module = importlib.import_module("ComfyUI-nunchaku.model_base.qwenimage")
                    QwenModelBase = qwen_base_module.NunchakuQwenImage
                    
                    patcher_module = importlib.import_module("ComfyUI-nunchaku.model_patcher")
                    NunchakuModelPatcher = patcher_module.NunchakuModelPatcher
                    
                  #  cstr("[Nunchaku Wrapper] ✓ Imported via importlib fallback").msg.print()
            
            #cstr(f"[Nunchaku Wrapper] ✓ ComfyFluxWrapper module: {ComfyFluxWrapper.__module__}").msg.print()
            cstr("[Nunchaku Wrapper] ✓ All Nunchaku classes imported successfully").msg.print()
        else:
            cstr("[Nunchaku Wrapper] ComfyUI-nunchaku extension not found").warning.print()
            ComfyFluxWrapper = None
            QwenConfig = None
            QwenModelBase = None
            NunchakuModelPatcher = None
            
    except Exception as e:
        import traceback
        cstr("[Nunchaku Wrapper] ERROR: Could not import Nunchaku classes:").error.print()
        cstr(f"[Nunchaku Wrapper] Exception type: {type(e).__name__}").error.print()
        cstr(f"[Nunchaku Wrapper] Exception message: {e}").error.print()
        traceback.print_exc()
        ComfyFluxWrapper = None
        QwenConfig = None
        QwenModelBase = None
        NunchakuModelPatcher = None
    
    NUNCHAKU_AVAILABLE = True
except ImportError as e:
    cstr(f"[Nunchaku Wrapper] Nunchaku package not available: {e}").warning.print()
    pass

# ComfyUI imports
try:
    import comfy.model_patcher
    import comfy.model_management
    import comfy.utils
    import comfy.model_detection
    from comfy.supported_models import Flux, FluxSchnell
except ImportError:
    # For standalone testing
    comfy = None


def is_nunchaku_model_by_name(filename: str) -> bool:
    # Detect Nunchaku models by filename patterns.
    #
    # Nunchaku quantized models typically follow naming conventions:
    # - svdq-fp4_* (SVDQuant FP4 quantization)
    # - svdq-int4_* (SVDQuant INT4 quantization)
    # - nunchaku-* (Generic Nunchaku prefix)
    # - *-quant-* (Generic quantization marker)
    #
    # Parameters
    # ----------
    # filename : str
    #     The model filename to check
    #
    # Returns
    # -------
    # bool
    #     True if filename matches Nunchaku patterns
    #
    # Examples
    # --------
    # >>> is_nunchaku_model_by_name("svdq-fp4_r32-flux-dev.safetensors")
    # True
    # >>> is_nunchaku_model_by_name("flux1-dev.safetensors")
    # False
    filename_lower = filename.lower()
    
    nunchaku_patterns = [
        'svdq-fp4',      # SVDQuant FP4
        'svdq-int4',     # SVDQuant INT4
        'nunchaku-',     # Nunchaku prefix
        'svdquant-',     # SVDQuant generic
        '-quant-',       # Generic quantization marker
    ]
    
    return any(pattern in filename_lower for pattern in nunchaku_patterns)


def is_nunchaku_model_by_metadata(model_path: str) -> bool:
    # Check if model is Nunchaku format by inspecting safetensors metadata.
    #
    # This reads only the metadata header, not the full weights, making it fast.
    # Nunchaku models store special metadata like 'comfy_config', 'quantization', etc.
    #
    # Parameters
    # ----------
    # model_path : str
    #     Full path to the model file
    #
    # Returns
    # -------
    # bool
    #     True if metadata indicates Nunchaku format
    if not os.path.exists(model_path):
        return False
    
    try:
        # Use safetensors to read only metadata (fast, no weight loading)
        import safetensors.torch
        
        with safetensors.safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
            
            if metadata is None:
                return False
            
            # Nunchaku models have these metadata keys
            nunchaku_metadata_keys = [
                'comfy_config',      # ComfyUI configuration embedded by Nunchaku
                'quantization',      # Quantization parameters
                'nunchaku_version',  # Version marker
                'svdquant',          # SVDQuant marker
            ]
            
            # Check if any Nunchaku-specific key exists
            for key in nunchaku_metadata_keys:
                if key in metadata:
                    return True
            
            # Also check metadata values for Nunchaku indicators
            metadata_str = str(metadata).lower()
            if 'nunchaku' in metadata_str or 'svdquant' in metadata_str:
                return True
    
    except Exception:
        # If metadata reading fails, assume not Nunchaku
        pass
    
    return False


def detect_nunchaku_model(model_path: str, model_name: str) -> bool:
    # Hybrid detection: Combine filename patterns and metadata checks.
    #
    # Uses a two-stage approach:
    # 1. Fast filename pattern matching (no I/O)
    # 2. Metadata inspection if filename check fails (minimal I/O)
    #
    # Parameters
    # ----------
    # model_path : str
    #     Full path to the model file
    # model_name : str
    #     Model filename (for pattern matching)
    #
    # Returns
    # -------
    # bool
    #     True if model is detected as Nunchaku format
    # Stage 1: Fast filename check
    if is_nunchaku_model_by_name(model_name):
        return True
    
    # Stage 2: Metadata check (if file exists and is accessible)
    if os.path.exists(model_path) and os.path.isfile(model_path):
        return is_nunchaku_model_by_metadata(model_path)
    
    return False


def load_nunchaku_model(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cpu_offload: bool = False,
    cache_threshold: float = 0.0,
    attention: str = "flash-attention2",
    data_type: str = "bfloat16",
    i2f_mode: str = "enabled",
    num_blocks_on_gpu: int = 1,
    use_pin_memory: bool = False,
    model_type: str = "flux"
) -> object:
    # Load a Nunchaku quantized model and wrap it for ComfyUI compatibility.
    #
    # Supports both Flux and Qwen-Image Nunchaku quantized models.
    #
    # Parameters
    # ----------
    # model_path : str
    #     Full path to the Nunchaku model file
    # device : torch.device, optional
    #     Device to load model on (default: auto-detect CUDA)
    # dtype : torch.dtype, optional
    #     Data type for model (default: based on data_type parameter)
    # cpu_offload : bool, default=False
    #     Enable CPU offload for low VRAM (<14GB)
    # cache_threshold : float, default=0.0
    #     First-block caching threshold (0=disabled, 0.12=typical)
    #     Higher values = faster but lower quality
    # attention : str, default="flash-attention2"
    #     Attention implementation: "flash-attention2" or "nunchaku-fp16"
    # data_type : str, default="bfloat16"
    #     Model data type: "bfloat16" (30/40-series) or "float16" (20-series GPUs)
    # i2f_mode : str, default="enabled"
    #     GEMM implementation for 20-series GPUs: "enabled" or "always"
    #     (ignored on other GPU architectures, Flux models only)
    # num_blocks_on_gpu : int, default=1
    #     Number of transformer blocks to keep on GPU when CPU offload is enabled
    #     (Qwen models only)
    # use_pin_memory : bool, default=False
    #     Use pinned memory for faster CPU-GPU transfer when CPU offload is enabled
    #     (Qwen models only)
    # model_type : str, default="flux"
    #     Model architecture type: "flux" for Flux models or "qwen" for Qwen-Image models
    #
    # Returns
    # -------
    # ModelPatcher
    #     ComfyUI ModelPatcher object (NunchakuModelPatcher for Qwen, standard for Flux)
    #
    # Raises
    # ------
    # RuntimeError
    #     If Nunchaku is not available or loading fails
    # ValueError
    #     If model file not found or invalid
    #
    # Examples
    # --------
    # >>> # Load Flux model
    # >>> model, name = load_nunchaku_model(
    # ...     "models/svdq-fp4_r32-flux.safetensors",
    # ...     cpu_offload=True,
    # ...     cache_threshold=0.12,
    # ...     model_type="flux"
    # ... )
    # >>>
    # >>> # Load Qwen model
    # >>> model, name = load_nunchaku_model(
    # ...     "models/qwen-image-quant.safetensors",
    # ...     cpu_offload=True,
    # ...     model_type="qwen"
    # ... )
    if not NUNCHAKU_AVAILABLE:
        raise RuntimeError(
            "Nunchaku support is not available.\n\n"
            "To load Nunchaku quantized models, please install ComfyUI-nunchaku:\n"
            "  1. Navigate to ComfyUI/custom_nodes/\n"
            "  2. Clone: git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku\n"
            "  3. Install: cd ComfyUI-nunchaku && pip install -r requirements.txt\n"
            "  4. Restart ComfyUI\n\n"
            "Alternatively, use a standard (non-quantized) model."
        )
    
    # Validate model type and check required components
    if model_type not in ["flux", "qwen"]:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'flux' or 'qwen'")
    
    if model_type == "flux" and ComfyFluxWrapper is None:
        raise RuntimeError(
            "Nunchaku ComfyFluxWrapper not found.\n"
            "Please ensure ComfyUI-nunchaku is properly installed."
        )
    
    if model_type == "qwen" and (QwenConfig is None or QwenModelBase is None or NunchakuModelPatcher is None):
        raise RuntimeError(
            "Nunchaku Qwen support not found.\n"
            "Please ensure ComfyUI-nunchaku is properly installed with Qwen model support."
        )
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    # Auto-detect device if not provided
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    # Auto-detect dtype if not provided, otherwise use data_type parameter
    if dtype is None:
        dtype = torch.float16 if data_type == "float16" else torch.bfloat16
    
    # Auto-enable CPU offload for low VRAM GPUs
    if cpu_offload == "auto" and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        cpu_offload = (gpu_memory_gb < 14)
    
    model_label = "Flux" if model_type == "flux" else "Qwen-Image"
    cstr(f"[Nunchaku {model_label}] Loading quantized model: {os.path.basename(model_path)}").msg.print()
    cstr(f"[Nunchaku {model_label}]   Device: {device}").msg.print()
    cstr(f"[Nunchaku {model_label}]   CPU Offload: {cpu_offload}").msg.print()
    
    if model_type == "flux":
        cstr(f"[Nunchaku {model_label}]   Data Type: {data_type} ({dtype})").msg.print()
        cstr(f"[Nunchaku {model_label}]   Cache Threshold: {cache_threshold}").msg.print()
        cstr(f"[Nunchaku {model_label}]   Attention: {attention}").msg.print()
        cstr(f"[Nunchaku {model_label}]   I2F Mode: {i2f_mode}").msg.print()
    else:  # qwen
        cstr(f"[Nunchaku {model_label}]   Num Blocks on GPU: {num_blocks_on_gpu}").msg.print()
        cstr(f"[Nunchaku {model_label}]   Use Pin Memory: {use_pin_memory}").msg.print()
    
    # ============================================================
    # Load model based on type
    # ============================================================
    
    if model_type == "qwen":
        # Qwen-Image models use state dict loading (no wrapper exists yet)
        # This is a simplified version that uses ComfyUI-nunchaku's model config
        import safetensors.torch
        import safetensors
        
        # Load state dict and metadata
        sd = safetensors.torch.load_file(model_path)
        
        # Read metadata from safetensors
        with safetensors.safe_open(model_path, framework="pt") as f:
            metadata = f.metadata() if f.metadata() else {}
        
        # Import utility functions from nunchaku package directly
        try:
            from nunchaku.utils import check_hardware_compatibility, get_precision_from_quantization_config
        except ImportError as e:
            raise ImportError(
                f"Failed to import nunchaku utility functions: {e}\n\n"
                "Make sure nunchaku package is properly installed:\n"
                "  pip install nunchaku\n"
            )
        
        # Parse quantization config from metadata
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        precision = get_precision_from_quantization_config(quantization_config)
        rank = quantization_config.get("rank", 32)
        
        # Check hardware compatibility
        check_hardware_compatibility(quantization_config, device)
        
        # Prepare state dict (handle checkpoint format)
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
        
        # Calculate parameters and dtype
        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd)
        
        # Type guard for QwenConfig
        if QwenConfig is None:
            raise ImportError("QwenConfig not available from ComfyUI-nunchaku")
        
        # Create model config (using globally imported QwenConfig)
        model_config = QwenConfig({
            "image_model": "qwen_image",
            "scale_shift": 0,
            "rank": rank,
            "precision": precision
        })
        model_config.optimizations["fp8"] = False
        
        # Determine dtype
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.scaled_fp8 is not None:
            weight_dtype = None
        
        if dtype is None:
            unet_dtype = comfy.model_management.unet_dtype(
                model_params=parameters,
                supported_dtypes=unet_weight_dtype,
                weight_dtype=weight_dtype
            )
        else:
            unet_dtype = dtype
        
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype, device, model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        
        # Create and load base model
        model = model_config.get_model(sd, "")
        offload_device = comfy.model_management.unet_offload_device()
        model = model.to(offload_device)
        model.load_model_weights(sd, "")
        
        # Get the Qwen transformer
        qwen_transformer = model.diffusion_model
        
        # Extract comfy_config from metadata if available
        comfy_config_str = metadata.get("comfy_config", None)
        if comfy_config_str:
            comfy_config = json.loads(comfy_config_str)
        else:
            # Fallback: use basic config
            comfy_config = {"model_config": {}}
        
        # Wrap transformer in ComfyQwenImageWrapper for LoRA support
        wrapper = ComfyQwenImageWrapper(
            qwen_transformer,
            comfy_config.get("model_config", {}),
            customized_forward=None,
            forward_kwargs={},
            cpu_offload_setting="auto" if cpu_offload else "disable",
            vram_margin_gb=4.0
        )
        
        # Set CPU offload if enabled
        if cpu_offload and hasattr(qwen_transformer, 'set_offload'):
            qwen_transformer.set_offload(
                True, 
                num_blocks_on_gpu=num_blocks_on_gpu,
                use_pin_memory=use_pin_memory
            )
        
        # Replace diffusion_model with wrapper
        model.diffusion_model = wrapper
        
        # Type guard for NunchakuModelPatcher
        if NunchakuModelPatcher is None:
            raise ImportError("NunchakuModelPatcher not available from ComfyUI-nunchaku")
        
        # Create model patcher (using globally imported NunchakuModelPatcher)
        model_patcher = NunchakuModelPatcher(
            model,
            load_device=device,
            offload_device=offload_device
        )
        
        cstr(f"[Nunchaku {model_label}] ✓ Model loaded successfully: {os.path.basename(model_path)}").msg.print()
        cstr(f"[Nunchaku {model_label}] ✓ Qwen LoRA support enabled via ComfyQwenImageWrapper").msg.print()
        
        return model_patcher
    
    else:  # model_type == "flux"
        # Type guards for Flux components
        if ComfyFluxWrapper is None:
            raise ImportError(
                "ComfyFluxWrapper not available.\n"
                "Make sure ComfyUI-nunchaku extension is properly installed.\n"
                "The wrappers/flux.py file should exist in ComfyUI-nunchaku."
            )
        
        if NunchakuFluxTransformer2dModel is None:
            raise ImportError("NunchakuFluxTransformer2dModel not available")
        
        # Load quantized Flux transformer
        transformer, metadata = NunchakuFluxTransformer2dModel.from_pretrained(  # type: ignore
            model_path,
            offload=cpu_offload,
            device=device,
            torch_dtype=dtype,
            return_metadata=True
        )
    
    # Apply caching if enabled
    if cache_threshold > 0:
        if apply_cache_on_transformer is None:
            raise ImportError("apply_cache_on_transformer not available")
        transformer = apply_cache_on_transformer(  # type: ignore
            transformer=transformer,
            residual_diff_threshold=cache_threshold
        )
    
    # Set attention implementation
    if attention == "nunchaku-fp16":
        transformer.set_attention_impl("nunchaku-fp16")
    else:
        transformer.set_attention_impl("flashattn2")
    
    # Extract ComfyUI config from metadata
    if metadata is None:
        raise ValueError("Model metadata not found - this may not be a valid Nunchaku model")
    
    comfy_config_str = metadata.get("comfy_config", None)
    if comfy_config_str is None:
        raise ValueError(
            "Model is missing 'comfy_config' metadata.\n"
            "This may be an older Nunchaku model or corrupted file."
        )
    
    comfy_config = json.loads(comfy_config_str)
    
    # Determine model class (Flux or FluxSchnell)
    model_class_name = comfy_config.get("model_class", "Flux")
    if model_class_name == "FluxSchnell":
        model_class = FluxSchnell
    else:
        model_class = Flux
    
    # Create ComfyUI model configuration
    model_config = model_class(comfy_config["model_config"])
    model_config.set_inference_dtype(dtype, None)
    model_config.custom_operations = None
    
    # Create ComfyUI model structure
    model = model_config.get_model({})
    
    # Wrap transformer in ComfyUI-compatible wrapper
    # After wrapping in ModelPatcher, PuLID will access this as: model_patcher.model.diffusion_model
    # PuLID-compatible signature: (transformer, config, pulid_pipeline, customized_forward, forward_kwargs)
    wrapper = ComfyFluxWrapper(
        transformer,
        comfy_config["model_config"],
        None,  # pulid_pipeline - set by PuLID loader later
        None,  # customized_forward - set by PuLID apply later
        {}     # forward_kwargs - empty dict
    )
    
    model.diffusion_model = wrapper
    
    # Create ModelPatcher for ComfyUI integration
    device_id = device.index if hasattr(device, 'index') else 0
    model_patcher = comfy.model_patcher.ModelPatcher(model, device, device_id)
    
    cstr(f"[Nunchaku {model_label}] ✓ Model loaded successfully: {os.path.basename(model_path)}").msg.print()
    
    return model_patcher


def get_nunchaku_info() -> dict:
    # Get information about Nunchaku support availability.
    #
    # Returns
    # -------
    # dict
    #     Dictionary with keys:
    #     - 'available': bool - Whether Nunchaku is available
    #     - 'version': str | None - Nunchaku version if available
    #     - 'wrapper_available': bool - Whether ComfyFluxWrapper is available
    info: dict[str, bool | str | None] = {
        'available': NUNCHAKU_AVAILABLE,
        'version': None,
        'wrapper_available': ComfyFluxWrapper is not None
    }
    
    if NUNCHAKU_AVAILABLE:
        try:
            import nunchaku
            # Try to get version from __version__ attribute
            version = getattr(nunchaku, '__version__', None)  # type: str | None
            
            # If __version__ not available, try package metadata
            if not version:
                try:
                    from importlib.metadata import version as get_version
                    version = get_version('nunchaku')
                except Exception:
                    pass
            
            info['version'] = version
        except Exception:
            pass
    
    return info


# ============================================================================
# Qwen LoRA Support - Composition Engine
# ============================================================================

# Centralized key mapping for Qwen LoRA layers
# This regex-based approach efficiently maps LoRA keys to model layer names
_QWEN_KEY_MAPPING = [
    # Fused QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._](q|k|v)[._]proj$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Fused Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._]qkv[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv", None),
    # Decomposed Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv", lambda m: m.group(3).upper()),
    # Fused QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Output Projections
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj[._]context$"), r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]add[._]out$"), r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out[._]0$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out", "regular", None),
    # Feed-Forward / MLP Layers (Standard)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]0(?:[._]proj)?$"), r"\1.\2.mlp_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]2$"), r"\1.\2.mlp_fc2", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]0(?:[._]proj)?$"),
     r"\1.\2.mlp_context_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]2$"), r"\1.\2.mlp_context_fc2", "regular", None),
    # Feed-Forward / MLP Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular", None),
    # Mod Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    # Single Block Projections
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]out$"), r"\1.\2.proj_out", "single_proj_out", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]mlp$"), r"\1.\2.mlp_fc1", "regular", None),
    # Normalization Layers
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]norm[._]linear$"), r"\1.\2.norm.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1[._]linear$"), r"\1.\2.norm1.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1_context[._]linear$"), r"\1.\2.norm1_context.linear",
     "regular", None),
    # Top-level diffusion_model modules
    (re.compile(r"^(img_in)$"), r"\1", "regular", None),
    (re.compile(r"^(txt_in)$"), r"\1", "regular", None),
    (re.compile(r"^(proj_out)$"), r"\1", "regular", None),
    (re.compile(r"^(norm_out)[._](linear)$"), r"\1.\2", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_1)$"), r"\1.\2.\3", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_2)$"), r"\1.\2.\3", "regular", None),
]

_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")


def _rename_layer_underscore_layer_name(old_name: str) -> str:
    # Convert underscore patterns to dot notation for Qwen layer names.
    rules = [
        (r'_(\d+)_attn_to_out_(\d+)', r'.\1.attn.to_out.\2'),
        (r'_(\d+)_img_mlp_net_(\d+)_proj', r'.\1.img_mlp.net.\2.proj'),
        (r'_(\d+)_txt_mlp_net_(\d+)_proj', r'.\1.txt_mlp.net.\2.proj'),
        (r'_(\d+)_img_mlp_net_(\d+)', r'.\1.img_mlp.net.\2'),
        (r'_(\d+)_txt_mlp_net_(\d+)', r'.\1.txt_mlp.net.\2'),
        (r'_(\d+)_img_mod_(\d+)', r'.\1.img_mod.\2'),
        (r'_(\d+)_txt_mod_(\d+)', r'.\1.txt_mod.\2'),
        (r'_(\d+)_attn_', r'.\1.attn.'),
    ]
    
    new_name = old_name
    for pattern, replacement in rules:
        new_name = re.sub(pattern, replacement, new_name)
    
    return new_name


def _classify_and_map_qwen_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    # Classify and map a Qwen LoRA key using regex patterns.
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer."):]
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
        k = _rename_layer_underscore_layer_name(k)
    
    base = None
    ab = None
    
    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")
        base = k[: m.start()]
        if "lora_A" in tag or tag.endswith(".A") or "down" in tag:
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B") or "up" in tag:
            ab = "B"
    else:
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]
    
    if base is None or ab is None:
        return None
    
    for pattern, template, group, comp_fn in _QWEN_KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab
    
    return None


def _is_indexable_module(m):
    # Check if a module is list-like.
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    # Traverse a path like 'a.b.3.c' to find and return a module.
    if not name:
        return model
    module = model
    for part in name.split("."):
        if not part:
            continue
        
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and _is_indexable_module(module):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                cstr(f"[Qwen LoRA] Failed to index module {name} with part {part}").warning.print()
                return None
        else:
            return None
    return module


def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    # Resolve a name string path to a module with fallback paths.
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m
    
    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module_by_name(model, alt)
        if m is not None:
            return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None:
            return alt, m
    
    mapping = {
        ".ff.net.0.proj": ".mlp_fc1", ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1", ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None:
                return alt, m
    
    # Debug output disabled for performance (only enable for troubleshooting)
    # cstr(f"[Qwen LoRA] Module not found: {name}").msg.print()
    return name, None


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Load LoRA state dict from path or return existing dict.
    if isinstance(lora_state_dict_or_path, (str, Path)):
        path = Path(lora_state_dict_or_path)
        if path.suffix == ".safetensors":
            try:
                from safetensors import safe_open
                state_dict = {}
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                return state_dict
            except ImportError:
                cstr("[Qwen LoRA] safetensors not available, falling back to torch.load").warning.print()
                return torch.load(path, map_location="cpu")
        else:
            return torch.load(path, map_location="cpu")
    return lora_state_dict_or_path


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Fuse Q/K/V LoRA weights into a single QKV tensor.
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None, None
    
    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]
    
    if not (A_q.shape == A_k.shape == A_v.shape):
        cstr(f"[Qwen LoRA] Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}").warning.print()
        return None, None, None
    
    alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")
    alpha_fused = None
    if alpha_q is not None and alpha_k is not None and alpha_v is not None and (
            alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q
    
    A_fused = torch.cat([A_q, A_k, A_v], dim=0)
    
    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
    B_fused[out_q + out_k:, 2 * r:] = B_v
    
    return A_fused, B_fused, alpha_fused


def _handle_proj_out_split(lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module) -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], List[str]]:
    # Split single-block proj_out LoRA into two branches.
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
    consumed: List[str] = []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed
    
    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed
    
    A_full, B_full, alpha = lora_dict[base_key].get("A"), lora_dict[base_key].get("B"), lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None:
        return result, consumed
    
    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if attn_to_out is None or mlp_fc2 is None or not hasattr(attn_to_out, "in_features") or not hasattr(mlp_fc2, "in_features"):
        return result, consumed
    
    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        cstr(f"[Qwen LoRA] {base_key}: A_full shape mismatch {A_full.shape} vs expected in_features {attn_in + mlp_in}").warning.print()
        return result, consumed
    
    A_attn, A_mlp = A_full[:, :attn_in], A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full.clone(), alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full.clone(), alpha)
    consumed.append(base_key)
    return result, consumed


def _qwen_unpack_lowrank_weight(weight: torch.Tensor, down: bool = True) -> torch.Tensor:
    # Unpack Nunchaku low-rank weight (placeholder - uses nunchaku if available).
    try:
        from nunchaku.lora.flux.nunchaku_converter import unpack_lowrank_weight
        return unpack_lowrank_weight(weight, down=down)
    except ImportError:
        # Fallback: assume already unpacked
        return weight


def _qwen_pack_lowrank_weight(weight: torch.Tensor, down: bool = True) -> torch.Tensor:
    # Pack Nunchaku low-rank weight (placeholder - uses nunchaku if available).
    try:
        from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight
        return pack_lowrank_weight(weight, down=down)
    except ImportError:
        # Fallback: assume packing not needed
        return weight


def _qwen_reorder_adanorm_lora_up(weight: torch.Tensor, splits: int = 6) -> torch.Tensor:
    # Reorder AdaNorm LoRA up weights (placeholder - uses nunchaku if available).
    try:
        from nunchaku.lora.flux.nunchaku_converter import reorder_adanorm_lora_up
        return reorder_adanorm_lora_up(weight, splits=splits)
    except ImportError:
        # Fallback: no reordering
        return weight


def _apply_lora_to_qwen_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str,
                                model: nn.Module) -> None:
    # Append combined LoRA weights to a Qwen module.
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")
    
    pd, pu = module.proj_down.data, module.proj_up.data
    pd = _qwen_unpack_lowrank_weight(pd, down=True)
    pu = _qwen_unpack_lowrank_weight(pu, down=False)
    
    base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]
    
    if pd.shape[1] == module.in_features:  # [rank, in]
        new_proj_down = torch.cat([pd, A], dim=0)
        axis_down = 0
    else:  # [in, rank]
        new_proj_down = torch.cat([pd, A.T], dim=1)
        axis_down = 1
    
    new_proj_up = torch.cat([pu, B], dim=1)
    
    module.proj_down.data = _qwen_pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = _qwen_pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]
    
    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down})
    slot["appended"] += A.shape[0]


def compose_qwen_loras_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> None:
    # Compose multiple LoRAs into a Qwen model with individual strengths.
    #
    # Parameters
    # ----------
    # model : torch.nn.Module
    #     The Qwen model to apply LoRAs to
    # lora_configs : List[Tuple[Union[str, Path, dict], float]]
    #     List of (lora_path_or_dict, strength) tuples
    cstr(f"[Qwen LoRA] Composing {len(lora_configs)} LoRAs for Qwen model...").msg.print()
    reset_qwen_lora_v2(model)
    
    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    # 1. Aggregate weights from all LoRAs
    for lora_path_or_dict, strength in lora_configs:
        lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
        lora_state_dict = _load_lora_state_dict(lora_path_or_dict)
        
        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in lora_state_dict.items():
            parsed = _classify_and_map_qwen_key(key)
            if parsed is None:
                continue
            
            group, base_key, comp, ab = parsed
            if group in ("qkv", "add_qkv") and comp is not None:
                lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value
        
        # Process grouped weights for this LoRA
        processed_groups = {}
        special_handled = set()
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled:
                continue
            
            if "qkv" in base_key:
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed_keys = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.update(consumed_keys)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")
            
            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)
        
        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append(
                {"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name})
    
    # 2. Apply aggregated weights to the model
    applied_modules_count = 0
    
    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None or not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
            continue
        
        all_A = []
        all_B_scaled = []
        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))
            
            if ".norm1.linear" in resolved_name or ".norm1_context.linear" in resolved_name:
                B = _qwen_reorder_adanorm_lora_up(B, splits=6)
            elif ".single_transformer_blocks." in resolved_name and ".norm.linear" in resolved_name:
                B = _qwen_reorder_adanorm_lora_up(B, splits=3)
            
            all_A.append(A.to(dtype=module.proj_down.dtype, device=module.proj_down.device))
            all_B_scaled.append((B * scale).to(dtype=module.proj_up.dtype, device=module.proj_up.device))
        
        if not all_A:
            continue
        
        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)
        
        _apply_lora_to_qwen_module(module, final_A, final_B, resolved_name, model)
        applied_modules_count += 1
    
    cstr(f"[Qwen LoRA] Applied LoRA compositions to {applied_modules_count} Qwen modules.").msg.print()


def reset_qwen_lora_v2(model: nn.Module) -> None:
    # Remove all appended LoRA weights from a Qwen model.
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return
    
    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue
        
        base_rank = info["base_rank"]
        with torch.no_grad():
            pd = _qwen_unpack_lowrank_weight(module.proj_down.data, down=True)
            pu = _qwen_unpack_lowrank_weight(module.proj_up.data, down=False)
            
            if info.get("axis_down", 0) == 0:  # [rank, in]
                pd_reset = pd[:base_rank, :].clone()
            else:  # [in, rank]
                pd_reset = pd[:, :base_rank].clone()
            pu_reset = pu[:, :base_rank].clone()
            
            module.proj_down.data = _qwen_pack_lowrank_weight(pd_reset, down=True)
            module.proj_up.data = _qwen_pack_lowrank_weight(pu_reset, down=False)
            module.rank = base_rank
    
    model._lora_slots.clear()
    model._lora_strength = 1.0
    cstr("[Qwen LoRA] All LoRA weights have been reset from the Qwen model.").msg.print()


# ============================================================================
# ComfyQwenImageWrapper - Qwen Model Wrapper with LoRA Support
# ============================================================================

class ComfyQwenImageWrapper(nn.Module):
    # Wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows with LoRA.
    #
    # This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    # It detects changes to its `loras` attribute and recomposes the underlying model
    # lazily when the forward pass is executed.
    #
    # Attributes
    # ----------
    # model : NunchakuQwenImageTransformer2DModel
    #     The wrapped Qwen transformer model
    # loras : List[Tuple[Union[str, Path, dict], float]]
    #     List of (lora_path, strength) tuples to apply
    # cpu_offload_setting : str
    #     CPU offload mode: "auto", "enable", or "disable"
    # vram_margin_gb : float
    #     VRAM safety margin for auto mode (default: 4.0 GB)
    
    def __init__(
            self,
            model: Any,  # NunchakuQwenImageTransformer2DModel
            config: dict,
            customized_forward: Optional[Callable] = None,
            forward_kwargs: Optional[dict] = None,
            cpu_offload_setting: str = "auto",
            vram_margin_gb: float = 4.0
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras: List[Tuple[Union[str, Path, dict], float]] = []
        self._applied_loras: Optional[List[Tuple[Union[str, Path, dict], float]]] = None
        
        self.cpu_offload_setting = cpu_offload_setting
        self.vram_margin_gb = vram_margin_gb
        
        cstr(f"Qwen CPU offload setting: '{cpu_offload_setting}' (VRAM margin: {vram_margin_gb}GB)").msg.print()
        
        self.customized_forward = customized_forward
        self.forward_kwargs = forward_kwargs or {}
        
        self._prev_timestep = None
        self._cache_context = None
        
        # Track last seen device to detect CPU/GPU moves
        self._last_device = None
    
    def to_safely(self, device):
        # Safely move the model to the specified device.
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(device)
        else:
            self.model.to(device)
        return self
    
    def forward(
            self,
            x,
            timestep,
            context=None,
            y=None,
            guidance=None,
            control=None,
            transformer_options=None,
            **kwargs,
    ):
        # Forward pass for the wrapped Qwen model.
        #
        # Detects changes to the `self.loras` list and recomposes the model
        # on-the-fly before inference.
        if transformer_options is None:
            transformer_options = {}
        
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            timestep_float = float(timestep)
        
        model_is_dirty = (
            not self.loras and
            hasattr(self.model, "_lora_slots") and self.model._lora_slots
        )
        
        # Deep comparison of LoRA stacks
        loras_changed = False
        if self._applied_loras is None or len(self._applied_loras) != len(self.loras):
            loras_changed = True
        else:
            for applied, current in zip(self._applied_loras, self.loras):
                if applied != current:
                    loras_changed = True
                    break
        
        # Detect device transition
        try:
            current_device = next(self.model.parameters()).device
        except Exception:
            current_device = None
        device_changed = (self._last_device != current_device)
        
        # Recompose LoRAs if needed
        if loras_changed or model_is_dirty or device_changed:
            reset_qwen_lora_v2(self.model)
            self._applied_loras = self.loras.copy()
            
            # Reset cache when LoRAs change
            if loras_changed:
                self._cache_context = None
                self._prev_timestep = None
                # Debug output disabled for performance
                # cstr("[Qwen LoRA] Cache reset due to LoRA change").msg.print()
            
            # Dynamic VRAM check for CPU offload
            offload_is_on = hasattr(self.model, "offload_manager") and self.model.offload_manager is not None
            should_enable_offload = offload_is_on
            
            if self.cpu_offload_setting == "auto" and not offload_is_on and self.loras:
                try:
                    import comfy.model_management
                    free_vram_gb = comfy.model_management.get_free_memory() / (1024 ** 3)
                    
                    if free_vram_gb < self.vram_margin_gb:
                        cstr(
                            f"[Qwen LoRA] Free VRAM is {free_vram_gb:.2f}GB (below safety margin of {self.vram_margin_gb}GB) "
                            f"and 'cpu_offload' is 'auto'. Force-enabling CPU offload for Qwen LoRA composition.").msg.print()
                        should_enable_offload = True
                    else:
                        cstr(
                            f"[Qwen LoRA] Free VRAM is {free_vram_gb:.2f}GB (>= {self.vram_margin_gb}GB margin). "
                            f"Qwen LoRAs will be composed without enabling CPU offload.").msg.print()
                except Exception as e:
                    cstr(f"[Qwen LoRA] Error during VRAM check for Qwen LoRA offloading: {e}").error.print()
            
            # Compose LoRAs
            compose_qwen_loras_v2(self.model, self.loras)
            
            # Validate composition
            try:
                has_slots = hasattr(self.model, "_lora_slots") and bool(self.model._lora_slots)
            except Exception:
                has_slots = True
            if self.loras and not has_slots:
                cstr("[Qwen LoRA] Qwen LoRA composition reported 0 target modules. Forcing reset and retry.").warning.print()
                try:
                    reset_qwen_lora_v2(self.model)
                    compose_qwen_loras_v2(self.model, self.loras)
                except Exception as e:
                    cstr(f"[Qwen LoRA] Qwen LoRA re-compose retry failed: {e}").error.print()
            
            # Rebuild offload manager if needed
            if should_enable_offload:
                if offload_is_on:
                    manager = self.model.offload_manager
                    offload_settings = {
                        "num_blocks_on_gpu": manager.num_blocks_on_gpu,
                        "use_pin_memory": manager.use_pin_memory,
                    }
                else:
                    offload_settings = {
                        "num_blocks_on_gpu": 1,
                        "use_pin_memory": False,
                    }
                    cstr("[Qwen LoRA] Building new CPU offload manager for Qwen due to LoRA VRAM check.").msg.print()
                
                self.model.set_offload(False)
                self.model.set_offload(True, **offload_settings)
            
            self._last_device = current_device
        
        # Execute model
        return self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
    
    def _execute_model(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        # Helper function to run the Qwen model's forward pass.
        model_device = next(self.model.parameters()).device
        
        # Move input tensors to model device
        if x.device != model_device:
            x = x.to(model_device)
        if context is not None and context.device != model_device:
            context = context.to(model_device)
        
        input_is_5d = x.ndim == 5
        if input_is_5d:
            x = x.squeeze(2)
        
        if self.customized_forward:
            with torch.inference_mode():
                return self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **self.forward_kwargs,
                    **kwargs,
                )
        else:
            with torch.inference_mode():
                if x.ndim == 4:
                    x = x.unsqueeze(2)
                
                return self.model(
                    x,
                    timestep,
                    context,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs,
                )
