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

# Florence-2 Wrapper for Smart LML
#
# This module provides graceful loading support for Florence-2 models with fallback.
# Handles import from comfyui-florence2 extension if available.
#
# Key Features:
# - Automatic detection of comfyui-florence2 extension
# - Graceful fallback to transformers AutoModel
# - Proper package path management
# - Support for custom model implementations

import sys
from typing import Optional, Any
from pathlib import Path
import torch

# Import cstr for consistent logging
from . import cstr

# Florence-2 availability flags
FLORENCE2_CUSTOM_AVAILABLE = False
Florence2ForConditionalGeneration: Optional[Any] = None
Florence2Config: Optional[Any] = None
Florence2Processor: Optional[Any] = None

# Try to import custom Florence-2 implementation
try:
    # Path: core/florence2_wrapper.py -> ComfyUI_Eclipse -> custom_nodes
    custom_nodes_path = Path(__file__).parent.parent.parent
    florence_path = custom_nodes_path / "comfyui-florence2"
    
    if florence_path.exists():
        # cstr("[Florence-2 Wrapper] Attempting to import custom Florence-2 classes...").msg.print()
        
        import importlib.util
        import types
        
        # Create a fake package to support relative imports
        fake_package_name = "comfyui_florence2_custom"
        fake_package = types.ModuleType(fake_package_name)
        fake_package.__path__ = [str(florence_path)]
        fake_package.__file__ = str(florence_path / "__init__.py")
        sys.modules[fake_package_name] = fake_package
        
        # Import configuration
        config_spec = importlib.util.spec_from_file_location(
            f"{fake_package_name}.configuration_florence2",
            florence_path / "configuration_florence2.py"
        )
        if config_spec and config_spec.loader:
            config_module = importlib.util.module_from_spec(config_spec)
            config_module.__package__ = fake_package_name
            sys.modules[f"{fake_package_name}.configuration_florence2"] = config_module
            config_spec.loader.exec_module(config_module)
            Florence2Config = config_module.Florence2Config
        
        # Import modeling
        modeling_spec = importlib.util.spec_from_file_location(
            f"{fake_package_name}.modeling_florence2",
            florence_path / "modeling_florence2.py"
        )
        if modeling_spec and modeling_spec.loader:
            modeling_module = importlib.util.module_from_spec(modeling_spec)
            modeling_module.__package__ = fake_package_name
            sys.modules[f"{fake_package_name}.modeling_florence2"] = modeling_module
            modeling_spec.loader.exec_module(modeling_module)
            Florence2ForConditionalGeneration = modeling_module.Florence2ForConditionalGeneration
        
        # Import processing (if it exists - some extensions may not have it)
        processing_file = florence_path / "processing_florence2.py"
        if processing_file.exists():
            processing_spec = importlib.util.spec_from_file_location(
                f"{fake_package_name}.processing_florence2",
                processing_file
            )
            if processing_spec and processing_spec.loader:
                processing_module = importlib.util.module_from_spec(processing_spec)
                processing_module.__package__ = fake_package_name
                sys.modules[f"{fake_package_name}.processing_florence2"] = processing_module
                processing_spec.loader.exec_module(processing_module)
                Florence2Processor = processing_module.Florence2Processor
        
        # Check if we got the essential classes
        if Florence2ForConditionalGeneration and Florence2Config:
            cstr("[Florence-2 Wrapper] ✓ Custom Florence-2 classes imported successfully").msg.print()
            FLORENCE2_CUSTOM_AVAILABLE = True
        else:
            cstr("[Florence-2 Wrapper] Custom Florence-2 classes incomplete, will use AutoModel").warning.print()
    else:
        cstr("[Florence-2 Wrapper] comfyui-florence2 extension not found, will use AutoModel").warning.print()
        
except Exception as e:
    import traceback
    error_msg = str(e)
    # Suppress the common "attempted relative import" error which is expected when extension isn't installed
    if "attempted relative import" not in error_msg:
        cstr(f"[Florence-2 Wrapper] Could not import custom Florence-2: {e}").warning.print()
    cstr("[Florence-2 Wrapper] Will fall back to transformers AutoModel").warning.print()
    # Don't print full traceback unless debugging
    # traceback.print_exc()


def load_florence2_model(model_path: str, **load_kwargs) -> Any:
    # Load Florence-2 model with custom implementation if available, fallback to AutoModel.
    # Supports both local model paths and HuggingFace repo IDs.
    #
    # Args:
    #     model_path: Path to local model directory or HuggingFace repo ID
    #     **load_kwargs: Additional arguments for from_pretrained (dtype, device_map, etc.)
    #
    # Returns:
    #     Loaded Florence-2 model
    # Determine if loading from local path or remote
    is_local = Path(model_path).exists()
    source = "local" if is_local else "remote"
    
    # CRITICAL: Resolve "auto" attention mode BEFORE trying any loading
    # Florence-2 doesn't support "auto" - must be resolved to specific mode
    requested_attn = load_kwargs.get('attn_implementation', 'auto')
    
    if requested_attn == 'auto':
        # Check if flash-attn is available
        try:
            import flash_attn
            load_kwargs['attn_implementation'] = 'flash_attention_2'
            requested_attn = 'flash_attention_2'
            cstr(f"[Florence-2 Wrapper] Auto mode: Selected flash_attention_2 (flash-attn available)").msg.print()
        except ImportError:
            # Fall back to sdpa (PyTorch built-in, good performance)
            load_kwargs['attn_implementation'] = 'sdpa'
            requested_attn = 'sdpa'
            cstr(f"[Florence-2 Wrapper] Auto mode: Selected sdpa (flash-attn not available)").msg.print()
    
    # Try custom implementation first
    if FLORENCE2_CUSTOM_AVAILABLE and Florence2ForConditionalGeneration:
        try:
            cstr(f"[Florence-2 Wrapper] Loading from {source} with custom implementation: {model_path}").msg.print()
            model = Florence2ForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=is_local,  # Prevent online lookup for local models
                **load_kwargs
            )
            # cstr(f"[Florence-2 Wrapper] ✓ Loaded with custom implementation").msg.print()
            return model
        except Exception as e:
            cstr(f"[Florence-2 Wrapper] Custom implementation failed: {e}").warning.print()
            cstr(f"[Florence-2 Wrapper] Falling back to AutoModel...").warning.print()
    
    # Fallback to AutoModel with trust_remote_code
    from transformers import AutoModelForCausalLM
    
    cstr(f"[Florence-2 Wrapper] Loading from {source} with AutoModelForCausalLM: {model_path}").msg.print()
    
    # Handle flash_attention_2 gracefully with fallback to sdpa
    # Some cached Florence-2 model code may not declare Flash Attention 2 support
    
    if requested_attn == 'flash_attention_2':
        try:
            cstr(f"[Florence-2 Wrapper] Attempting Flash Attention 2...").msg.print()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local,
                **load_kwargs
            )
            cstr(f"[Florence-2 Wrapper] ✓ Loaded with Flash Attention 2").msg.print()
            return model
        except (ValueError, ImportError) as e:
            if "does not support Flash Attention 2.0" in str(e) or "flash_attn" in str(e):
                cstr(f"[Florence-2 Wrapper] Flash Attention 2 not supported by cached model code").warning.print()
                cstr(f"[Florence-2 Wrapper] Your Florence-2 model uses outdated cached code from HuggingFace").error.print()
                
                # Extract cache path from model_path for user guidance
                import os
                cache_hint = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules")
                model_name = Path(model_path).name if is_local else model_path.split('/')[-1]
                
                cstr(f"[Florence-2 Wrapper] To update: Delete cached folder and restart ComfyUI:").error.print()
                cstr(f"[Florence-2 Wrapper]   Location: {cache_hint}\\{model_name.replace('/', '_').replace('-', '_hyphen_')}").error.print()
                cstr(f"[Florence-2 Wrapper]   Or run: Remove-Item '{cache_hint}\\{model_name.replace('/', '_').replace('-', '_hyphen_')}' -Recurse -Force").error.print()
                cstr(f"[Florence-2 Wrapper] Falling back to SDPA (still faster than eager mode)").warning.print()
                
                load_kwargs['attn_implementation'] = 'sdpa'
            else:
                raise
    
    # Load with requested attention mode (or fallback to sdpa)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local,  # Prevent online lookup for local models
        **load_kwargs
    )
    
    # Add _supports_sdpa to model class if not present (custom models from HF may lack it)
    if not hasattr(type(model), '_supports_sdpa'):
        cstr(f"[Florence-2 Wrapper] Adding _supports_sdpa=True to {type(model).__name__}").warning.print()
        type(model)._supports_sdpa = True
    
    # Also patch language_model subcomponent if it exists (for Florence2ForConditionalGeneration)
    if hasattr(model, 'language_model') and not hasattr(type(model.language_model), '_supports_sdpa'):
        cstr(f"[Florence-2 Wrapper] Adding _supports_sdpa=True to {type(model.language_model).__name__}").warning.print()
        type(model.language_model)._supports_sdpa = True
    
    attn_used = load_kwargs.get('attn_implementation', 'auto')
    cstr(f"[Florence-2 Wrapper] ✓ Loaded with AutoModel from {source}, attention={attn_used}").msg.print()
    return model


def load_florence2_processor(model_path: str, **kwargs) -> Any:
    # Load Florence-2 processor with custom implementation if available, fallback to AutoProcessor.
    # Supports both local model paths and HuggingFace repo IDs.
    #
    # Args:
    #     model_path: Path to local model directory or HuggingFace repo ID
    #     **kwargs: Additional arguments for from_pretrained
    #
    # Returns:
    #     Loaded Florence-2 processor
    # Determine if loading from local path or remote
    is_local = Path(model_path).exists()
    
    # Try custom processor first
    if FLORENCE2_CUSTOM_AVAILABLE and Florence2Processor:
        try:
            processor = Florence2Processor.from_pretrained(
                model_path,
                local_files_only=is_local,
                **kwargs
            )
            return processor
        except Exception as e:
            cstr(f"[Florence-2 Wrapper] Custom processor failed: {e}, using AutoProcessor").warning.print()
    
    # Fallback to AutoProcessor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local,  # Prevent online lookup for local models
        **kwargs
    )
    return processor


# Export public API
__all__ = [
    'FLORENCE2_CUSTOM_AVAILABLE',
    'Florence2ForConditionalGeneration',
    'Florence2Config',
    'Florence2Processor',
    'load_florence2_model',
    'load_florence2_processor',
]
