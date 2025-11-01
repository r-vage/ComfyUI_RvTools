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

"""
Nunchaku Model Wrapper for Smart Loader Plus

This module provides detection and loading support for Nunchaku quantized models.
Nunchaku models are quantized FLUX models (INT4/FP4/FP8) that require special loading.

Key Features:
- Automatic detection via filename patterns and metadata inspection
- Graceful fallback when ComfyUI-nunchaku is not installed
- Hybrid detection: filename patterns + metadata checks
- Compatible with ComfyUI ModelPatcher interface
"""

import os
import json
from typing import Optional, Any, TYPE_CHECKING
import torch

#print("[Nunchaku Wrapper] Module loading started...")

# Try to import Nunchaku - graceful fallback if not available
NUNCHAKU_AVAILABLE = False
NunchakuFluxTransformer2dModel: Optional[Any] = None
NunchakuQwenImageTransformer2DModel: Optional[Any] = None
apply_cache_on_transformer: Optional[Any] = None
ComfyFluxWrapper: Optional[Any] = None
QwenConfig: Optional[Any] = None
QwenModelBase: Optional[Any] = None
NunchakuModelPatcher: Optional[Any] = None

#print("[Nunchaku Wrapper] Starting Nunchaku imports...")

try:
    from nunchaku import NunchakuFluxTransformer2dModel as _NunchakuFluxTransformer2dModel
    from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer as _apply_cache_on_transformer
    
    NunchakuFluxTransformer2dModel = _NunchakuFluxTransformer2dModel
    apply_cache_on_transformer = _apply_cache_on_transformer
    
    print("[Nunchaku Wrapper] ✓ Nunchaku base imports successful")
    
    # Try to import Qwen model
    try:
        from nunchaku.models.qwenimage import NunchakuQwenImageTransformer2DModel as _NunchakuQwenImageTransformer2DModel
        NunchakuQwenImageTransformer2DModel = _NunchakuQwenImageTransformer2DModel
        #print("[Nunchaku Wrapper] ✓ Qwen model import successful")
    except ImportError as e:
        #print(f"[Nunchaku Wrapper] Qwen model not available: {e}")
        NunchakuQwenImageTransformer2DModel = None
    
    # Import ComfyFluxWrapper and Qwen classes from ComfyUI-nunchaku extension
    try:
        import sys
        from pathlib import Path
        
        # Look for ComfyUI-nunchaku in custom_nodes
        custom_nodes_path = Path(__file__).parent.parent.parent.parent
        nunchaku_path = custom_nodes_path / "ComfyUI-nunchaku"
        
        #print(f"[Nunchaku Wrapper] Looking for ComfyUI-nunchaku at: {nunchaku_path}")
        #print(f"[Nunchaku Wrapper] Path exists: {nunchaku_path.exists()}")
        
        if nunchaku_path.exists():
            # Import classes using the EXACT same module path that PuLID uses
            # PuLID's relative imports create entries like 'D:\...\ComfyUI-nunchaku.wrappers.flux' in sys.modules
            # We need to check sys.modules to find the actual module name and reuse it
            print("[Nunchaku Wrapper] Attempting to import Nunchaku classes...")
            
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
                    
                  #  print(f"[Nunchaku Wrapper] ✓ Imported via importlib fallback")
            
            #print(f"[Nunchaku Wrapper] ✓ ComfyFluxWrapper module: {ComfyFluxWrapper.__module__}")
            print(f"[Nunchaku Wrapper] ✓ All Nunchaku classes imported successfully")
        else:
            raise ImportError("ComfyUI-nunchaku not found")
            
    except Exception as e:
        import traceback
        print(f"[Nunchaku Wrapper] ERROR: Could not import Nunchaku classes:")
        print(f"[Nunchaku Wrapper] Exception type: {type(e).__name__}")
        print(f"[Nunchaku Wrapper] Exception message: {e}")
        traceback.print_exc()
        ComfyFluxWrapper = None
        QwenConfig = None
        QwenModelBase = None
        NunchakuModelPatcher = None
    
    NUNCHAKU_AVAILABLE = True
except ImportError as e:
    print(f"[Nunchaku Wrapper] Nunchaku package not available: {e}")
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
    """
    Detect Nunchaku models by filename patterns.
    
    Nunchaku quantized models typically follow naming conventions:
    - svdq-fp4_* (SVDQuant FP4 quantization)
    - svdq-int4_* (SVDQuant INT4 quantization)
    - nunchaku-* (Generic Nunchaku prefix)
    - *-quant-* (Generic quantization marker)
    
    Parameters
    ----------
    filename : str
        The model filename to check
        
    Returns
    -------
    bool
        True if filename matches Nunchaku patterns
        
    Examples
    --------
    >>> is_nunchaku_model_by_name("svdq-fp4_r32-flux-dev.safetensors")
    True
    >>> is_nunchaku_model_by_name("flux1-dev.safetensors")
    False
    """
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
    """
    Check if model is Nunchaku format by inspecting safetensors metadata.
    
    This reads only the metadata header, not the full weights, making it fast.
    Nunchaku models store special metadata like 'comfy_config', 'quantization', etc.
    
    Parameters
    ----------
    model_path : str
        Full path to the model file
        
    Returns
    -------
    bool
        True if metadata indicates Nunchaku format
    """
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
    """
    Hybrid detection: Combine filename patterns and metadata checks.
    
    Uses a two-stage approach:
    1. Fast filename pattern matching (no I/O)
    2. Metadata inspection if filename check fails (minimal I/O)
    
    Parameters
    ----------
    model_path : str
        Full path to the model file
    model_name : str
        Model filename (for pattern matching)
        
    Returns
    -------
    bool
        True if model is detected as Nunchaku format
    """
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
    """
    Load a Nunchaku quantized model and wrap it for ComfyUI compatibility.
    
    Supports both Flux and Qwen-Image Nunchaku quantized models.
    
    Parameters
    ----------
    model_path : str
        Full path to the Nunchaku model file
    device : torch.device, optional
        Device to load model on (default: auto-detect CUDA)
    dtype : torch.dtype, optional
        Data type for model (default: based on data_type parameter)
    cpu_offload : bool, default=False
        Enable CPU offload for low VRAM (<14GB)
    cache_threshold : float, default=0.0
        First-block caching threshold (0=disabled, 0.12=typical)
        Higher values = faster but lower quality
    attention : str, default="flash-attention2"
        Attention implementation: "flash-attention2" or "nunchaku-fp16"
    data_type : str, default="bfloat16"
        Model data type: "bfloat16" (30/40-series) or "float16" (20-series GPUs)
    i2f_mode : str, default="enabled"
        GEMM implementation for 20-series GPUs: "enabled" or "always"
        (ignored on other GPU architectures, Flux models only)
    num_blocks_on_gpu : int, default=1
        Number of transformer blocks to keep on GPU when CPU offload is enabled
        (Qwen models only)
    use_pin_memory : bool, default=False
        Use pinned memory for faster CPU-GPU transfer when CPU offload is enabled
        (Qwen models only)
    model_type : str, default="flux"
        Model architecture type: "flux" for Flux models or "qwen" for Qwen-Image models
        
    Returns
    -------
    ModelPatcher
        ComfyUI ModelPatcher object (NunchakuModelPatcher for Qwen, standard for Flux)
        
    Raises
    ------
    RuntimeError
        If Nunchaku is not available or loading fails
    ValueError
        If model file not found or invalid
        
    Examples
    --------
    >>> # Load Flux model
    >>> model, name = load_nunchaku_model(
    ...     "models/svdq-fp4_r32-flux.safetensors",
    ...     cpu_offload=True,
    ...     cache_threshold=0.12,
    ...     model_type="flux"
    ... )
    >>> 
    >>> # Load Qwen model
    >>> model, name = load_nunchaku_model(
    ...     "models/qwen-image-quant.safetensors",
    ...     cpu_offload=True,
    ...     model_type="qwen"
    ... )
    """
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
    print(f"[Nunchaku {model_label}] Loading quantized model: {os.path.basename(model_path)}")
    print(f"[Nunchaku {model_label}]   Device: {device}")
    print(f"[Nunchaku {model_label}]   CPU Offload: {cpu_offload}")
    
    if model_type == "flux":
        print(f"[Nunchaku {model_label}]   Data Type: {data_type} ({dtype})")
        print(f"[Nunchaku {model_label}]   Cache Threshold: {cache_threshold}")
        print(f"[Nunchaku {model_label}]   Attention: {attention}")
        print(f"[Nunchaku {model_label}]   I2F Mode: {i2f_mode}")
    else:  # qwen
        print(f"[Nunchaku {model_label}]   Num Blocks on GPU: {num_blocks_on_gpu}")
        print(f"[Nunchaku {model_label}]   Use Pin Memory: {use_pin_memory}")
    
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
        
        # Create and load model
        model = model_config.get_model(sd, "")
        offload_device = comfy.model_management.unet_offload_device()
        model = model.to(offload_device)
        model.load_model_weights(sd, "")
        
        # Type guard for NunchakuModelPatcher
        if NunchakuModelPatcher is None:
            raise ImportError("NunchakuModelPatcher not available from ComfyUI-nunchaku")
        
        # Create model patcher (using globally imported NunchakuModelPatcher)
        model_patcher = NunchakuModelPatcher(
            model,
            load_device=device,
            offload_device=offload_device
        )
        
        # Set CPU offload if enabled
        if cpu_offload and hasattr(model_patcher.model, 'set_offload'):
            model_patcher.model.set_offload(
                True, 
                num_blocks_on_gpu=num_blocks_on_gpu,
                use_pin_memory=use_pin_memory
            )
        
        print(f"[Nunchaku {model_label}] ✓ Model loaded successfully: {os.path.basename(model_path)}")
        
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
    
    print(f"[Nunchaku {model_label}] ✓ Model loaded successfully: {os.path.basename(model_path)}")
    
    return model_patcher


def get_nunchaku_info() -> dict:
    """
    Get information about Nunchaku support availability.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'available': bool - Whether Nunchaku is available
        - 'version': str | None - Nunchaku version if available
        - 'wrapper_available': bool - Whether ComfyFluxWrapper is available
    """
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
