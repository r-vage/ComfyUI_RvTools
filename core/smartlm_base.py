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

# Smart LML Base - Unified Vision-Language Model Framework
#
# Supports multiple VLM architectures:
# - QwenVL (Qwen3-VL, Qwen2.5-VL) - Advanced multimodal with video support
# - Florence-2 - Fast captioning and prompt generation
# - Extensible for future models (LLaVA, InternVL, etc.)
#
# Template-based configuration system with auto-detection of model type.

import json
import gc
import torch
import numpy as np
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from PIL import Image
from enum import Enum

# Import cstr for logging
from . import cstr
from . import smartlm_florence2
from . import smartlm_qwenvl
from . import smartlm_llm
from .smartlm_files import download_with_progress, get_llm_model_list, get_mmproj_list, calculate_model_size, search_model_file, extract_repo_id_from_url, verify_model_integrity, calculate_file_hash

# Model type detection
class ModelType(Enum):
    QWENVL = "qwenvl"
    FLORENCE2 = "florence2"
    LLM = "llm"  # Text-only LLM (no vision)
    UNKNOWN = "unknown"

# Base paths
NODE_DIR = Path(__file__).parent.parent
REPO_TEMPLATE_DIR = NODE_DIR / "templates" / "smartlm_templates"

# Primary location: models/Eclipse/smartlm_templates (user-editable)
import folder_paths
ECLIPSE_TEMPLATE_DIR = Path(folder_paths.models_dir) / "Eclipse" / "smartlm_templates"

def get_template_dir():
    """Get current template directory (dynamic - always prefer Eclipse if exists)."""
    return ECLIPSE_TEMPLATE_DIR if ECLIPSE_TEMPLATE_DIR.exists() else REPO_TEMPLATE_DIR

# Use Eclipse folder if available, otherwise fallback to repo
TEMPLATE_DIR = get_template_dir()

# Config files: check Eclipse first, fallback to repo
ECLIPSE_CONFIG_DIR = Path(folder_paths.models_dir) / "Eclipse" / "config"
REPO_CONFIG_DIR = NODE_DIR / "templates" / "config"

# Always load smartlm_prompt_defaults.json from repo (no user edits needed)
PROMPT_CONFIG_PATH = REPO_CONFIG_DIR / "smartlm_prompt_defaults.json"
LLM_FEW_SHOT_CONFIG_PATH = (ECLIPSE_CONFIG_DIR / "llm_few_shot_training.json" 
                            if (ECLIPSE_CONFIG_DIR / "llm_few_shot_training.json").exists() 
                            else REPO_CONFIG_DIR / "llm_few_shot_training.json")

# Ensure template directory exists (prefer Eclipse location)
ECLIPSE_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# Global configuration storage
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}
LLM_FEW_SHOT_EXAMPLES = {}  # LLM few-shot training examples
LLAMA_CPP_AVAILABLE = False  # Will be set during initialization
LLAMA_CPP_MODULE = None  # Will be "llama_cpp_cuda" or "llama_cpp"

def is_model_architecture_supported(repo_id: str) -> bool:
    """Check if a model architecture is supported by the installed transformers version."""
    if not repo_id:
        return True  # No repo_id means likely GGUF or local - assume supported
    
    # Known unsupported architectures and their minimum required transformers versions
    # Format: {pattern: (min_version, description)}
    architecture_requirements = {
        'qwen3-vl': ('4.57.1', 'Qwen3-VL'),  # Qwen3-VL requires transformers >= 4.57.1
        'qwen3_vl': ('4.57.1', 'Qwen3-VL'),
    }
    
    # Check if repo contains known unsupported architecture markers
    repo_lower = repo_id.lower()
    unsupported_arch = None
    
    for pattern, (min_version, arch_name) in architecture_requirements.items():
        # More specific pattern matching
        if pattern in repo_lower:
            unsupported_arch = (pattern, min_version, arch_name)
            break
    
    if not unsupported_arch:
        return True  # Not a known problematic architecture
    
    # Check transformers version
    try:
        import transformers
        from packaging import version
        
        current_version = version.parse(transformers.__version__)
        required_version = version.parse(unsupported_arch[1])
        
        return current_version >= required_version
    except:
        # If we can't check version, assume supported to avoid false positives
        return True


def get_template_list() -> List[str]:
    """Get list of available Smart LML templates, filtering out unsupported architectures."""
    templates = []  # Start with empty list, will add "None" at the top after sorting
    filtered_templates = []
    template_dir = get_template_dir()
    if template_dir.exists():
        for f in template_dir.iterdir():
            if f.suffix == '.json' and not f.name.startswith('_'):
                # Check if template's model architecture is supported
                try:
                    with open(f, 'r') as file:
                        template_data = json.load(file)
                        repo_id = template_data.get('repo_id', '')
                        
                        # Filter out unsupported architectures
                        is_supported = is_model_architecture_supported(repo_id)
                        if is_supported:
                            templates.append(f.stem)
                        else:
                            filtered_templates.append(f.stem)
                except Exception as e:
                    # If we can't read the template, include it anyway
                    templates.append(f.stem)
    
    # Log filtered templates summary at startup (only once)
    if filtered_templates and not hasattr(get_template_list, '_logged'):
        get_template_list._logged = True
        
        # Group templates by base model to reduce output clutter
        # Extract base model names (e.g., "Qwen3-VL-2B-Instruct" -> "Qwen3-VL")
        base_models = set()
        for template in filtered_templates:
            # Common patterns: Model-Size-Variant, Model-Variant-Size
            template_lower = template.lower()
            if 'qwen3-vl' in template_lower or 'qwen3_vl' in template_lower:
                base_models.add('Qwen3-VL')
            # Add more patterns as needed
        
        if base_models:
            models_str = ', '.join(sorted(base_models))
            cstr(f"[SmartLM] {len(filtered_templates)} template(s) hidden (require transformers >= 4.57.1): {models_str} variants").warning.print()
        else:
            # Fallback to listing all if grouping fails
            cstr(f"[SmartLM] {len(filtered_templates)} template(s) hidden (require newer transformers): {', '.join(filtered_templates)}").warning.print()
    
    # Sort templates alphabetically, then add "None" at the top for reset functionality
    sorted_templates = sorted(templates) if templates else ["Qwen2.5-VL-3B-Instruct"]
    return ["None"] + sorted_templates



def load_template(name: str) -> dict:
    # Load a Smart LML template configuration
    if not name or name == "None":
        return {}
    template_path = get_template_dir() / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        cstr(f"[SmartLM] Error loading template {name}: {e}").error.print()
    return {}

def update_template_local_path(name: str, local_path: str) -> bool:
    # Update template's local_path field after successful download for offline usage
    if not name or name == "None" or name == "__temp_manual_config__":
        return False
    template_path = get_template_dir() / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Compare paths directly (preserve trailing slashes for directories)
            current_path = template_data.get("local_path") or ""
            
            # Only update if local_path is empty or different
            if current_path != local_path:
                template_data["local_path"] = local_path
                with open(template_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                cstr(f"[SmartLM] ✓ Updated template '{name}' with local_path: {local_path}").msg.print()
                return True
    except Exception as e:
        cstr(f"[SmartLM] Warning: Could not update template {name}: {e}").warning.print()
    return False

def update_template_vram_requirement(name: str, vram_req: dict) -> bool:
    # Update template's vram_requirement with actual file size after download
    if not name or name == "None" or name == "__temp_manual_config__":
        return False
    template_path = get_template_dir() / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Only update if different
            current_vram = template_data.get("vram_requirement", {})
            if current_vram != vram_req:
                template_data["vram_requirement"] = vram_req
                with open(template_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                return True
    except Exception as e:
        cstr(f"[SmartLM] Warning: Could not update template {name} vram_requirement: {e}").warning.print()
    return False

def update_template_mmproj_path(name: str, mmproj_path: str) -> bool:
    # Update template's mmproj_path field after successful download
    if not name or name == "None" or name == "__temp_manual_config__":
        return False
    template_path = get_template_dir() / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            current_path = template_data.get("mmproj_path") or ""
            
            # Only update if different
            if current_path != mmproj_path:
                template_data["mmproj_path"] = mmproj_path
                with open(template_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                return True
    except Exception as e:
        cstr(f"[SmartLM] Warning: Could not update template {name} mmproj_path: {e}").warning.print()
    return False

def update_template_settings(name: str, settings: dict, auto_save: bool = True) -> bool:
    # Update template with changed widget settings if auto_save is enabled
    #
    # Args:
    #     name: Template name
    #     settings: Dict of changed settings to update (e.g., {"max_tokens": 1024, "florence_task": "caption"})
    #     auto_save: Whether to save changes (user preference)
    #
    # Returns:
    #     True if template was updated, False otherwise
    if not auto_save or not name or name == "None" or not settings:
        return False
    
    template_path = get_template_dir() / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Track what changed
            changes = []
            for key, value in settings.items():
                if template_data.get(key) != value:
                    template_data[key] = value
                    changes.append(f"{key}={value}")
            
            # Save if anything changed
            if changes:
                with open(template_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                cstr(f"[SmartLM] ✓ Auto-saved template '{name}': {', '.join(changes)}").msg.print()
                return True
    except Exception as e:
        cstr(f"[SmartLM] Warning: Could not auto-save template {name}: {e}").warning.print()
    return False

def detect_model_type(template_info: dict) -> ModelType:
    # Detect model type from template configuration
    model_type = template_info.get("model_type", "").lower()
    repo_id = template_info.get("repo_id", "").lower()
    local_path = template_info.get("local_path", "").lower()
    mmproj_path = template_info.get("mmproj_path", "")
    
    # Explicit LLM type
    if model_type == "llm":
        return ModelType.LLM
    
    # QwenVL detection
    if model_type == "qwenvl" or "qwen" in repo_id:
        return ModelType.QWENVL
    
    # Florence-2 detection
    if model_type == "florence2" or "florence" in repo_id:
        return ModelType.FLORENCE2
    
    # Text-only GGUF detection: GGUF file without mmproj
    if local_path.endswith(".gguf") and not mmproj_path:
        return ModelType.LLM
    
    return ModelType.UNKNOWN

def load_prompt_configs():
    # Load prompt configurations for all model types
    global MODEL_CONFIGS, SYSTEM_PROMPTS, LLM_FEW_SHOT_EXAMPLES
    
    try:
        with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
            # QwenVL prompts
            SYSTEM_PROMPTS = config_data.get("_system_prompts", {})
            MODEL_CONFIGS["_preset_prompts"] = config_data.get("_preset_prompts", [])
            
            # Florence-2 tasks (for dropdown population - actual prompts are hardcoded in generation)
            smartlm_florence2.update_florence_tasks(config_data.get("_florence2_tasks", smartlm_florence2.FLORENCE_TASKS))
            
        florence_tasks = smartlm_florence2.get_florence_tasks()
        cstr(f"[SmartLM] Loaded {len(SYSTEM_PROMPTS)} QwenVL prompts, {len(florence_tasks)} Florence-2 tasks").msg.print()
    except Exception as exc:
        cstr(f"[SmartLM] Config load failed: {exc}").error.print()
        SYSTEM_PROMPTS = {}
        MISTRAL3_TASKS = {}
        MODEL_CONFIGS["_preset_prompts"] = []
    
    # Load LLM few-shot training examples
    try:
        with open(LLM_FEW_SHOT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            LLM_FEW_SHOT_EXAMPLES = json.load(f)
        cstr(f"[SmartLM] Loaded LLM few-shot training examples").msg.print()
    except Exception as exc:
        cstr(f"[SmartLM] LLM few-shot config load failed: {exc}").warning.print()
        # Fallback to minimal defaults
        LLM_FEW_SHOT_EXAMPLES = {
            "prompt_generation": {
                "system_prompt": "You are a helpful assistant.",
                "examples": []
            },
            "direct_chat": {
                "system_prompt": "You are a helpful assistant. Try your best to give the best response possible to the user.",
                "examples": []
            }
        }
    
    # Load templates
    template_count = len(get_template_list())
    cstr(f"[SmartLM] Found {template_count} model templates").msg.print()
    
    # Show transformers version
    try:
        import transformers
        cstr(f"[SmartLM] Transformers version: {transformers.__version__}").msg.print()
    except Exception:
        cstr(f"[SmartLM] Transformers not found").warning.print()
    
    # Show llama-cpp-python version and store availability
    # Check for llama-cpp-python (has integrated CUDA support since 0.3.x)
    global LLAMA_CPP_AVAILABLE, LLAMA_CPP_MODULE
    LLAMA_CPP_AVAILABLE = False
    LLAMA_CPP_MODULE = None
    try:
        import llama_cpp
        LLAMA_CPP_MODULE = "llama_cpp"
        LLAMA_CPP_AVAILABLE = True
        version = getattr(llama_cpp, '__version__', 'unknown')
        cstr(f"[SmartLM] llama-cpp-python version: {version}").msg.print()
        
        # Check if GPU offloading is supported
        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
            try:
                gpu_support = llama_cpp.llama_supports_gpu_offload()
                if gpu_support:
                    cstr(f"[SmartLM] ✓ GPU offloading available").msg.print()
            except:
                pass
    except ImportError:
        cstr(f"[SmartLM] llama-cpp-python not found (optional for GGUF models)").msg.print()

# Initialize on import
if not MODEL_CONFIGS:
    load_prompt_configs()

class SmartLMLBase:
    # Base class for unified vision-language model handling
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_template = None
        self.model_type = ModelType.UNKNOWN
        self.device_info = self.get_device_info()
    
    def clear(self):
        # Clear loaded model from memory
        # CRITICAL: Clean up chat_handler FIRST before model (it's the main VRAM holder)
        if hasattr(self, 'chat_handler_ref') and self.chat_handler_ref is not None:
            try:
                cstr("[SmartLM] Freeing chat_handler (mtmd/CLIP context)...").msg.print()
                chat_handler = self.chat_handler_ref
                
                # NEW VISION SYSTEM: Free mtmd context (Qwen2.5-VL uses this)
                if hasattr(chat_handler, 'mtmd_ctx') and chat_handler.mtmd_ctx is not None:
                    cstr("[SmartLM] Freeing mtmd vision context...").msg.print()
                    try:
                        # Call _exit_stack to trigger mtmd_free callback
                        if hasattr(chat_handler, '_exit_stack'):
                            chat_handler._exit_stack.close()
                        # Or call mtmd_free directly
                        elif hasattr(chat_handler, '_mtmd_cpp'):
                            chat_handler._mtmd_cpp.mtmd_free(chat_handler.mtmd_ctx)
                            chat_handler.mtmd_ctx = None
                    except Exception as e:
                        cstr(f"[SmartLM] mtmd_free error (may be ok): {e}").msg.print()
                
                # OLD VISION SYSTEM: Try to delete CLIP model directly (legacy LLaVA)
                if hasattr(chat_handler, 'clip_model') and chat_handler.clip_model is not None:
                    cstr("[SmartLM] Deleting legacy CLIP model...").msg.print()
                    del chat_handler.clip_model
                    chat_handler.clip_model = None
                
                # Clear any cached embeddings
                if hasattr(chat_handler, 'image_embeds'):
                    chat_handler.image_embeds = None
                if hasattr(chat_handler, '_image_embeds'):
                    chat_handler._image_embeds = None
                if hasattr(chat_handler, 'embeds'):
                    chat_handler.embeds = None
                if hasattr(chat_handler, '_last_image_embed'):
                    chat_handler._last_image_embed = None
                if hasattr(chat_handler, '_cache'):
                    chat_handler._cache.clear()
                
                # Call cleanup method if available
                if hasattr(chat_handler, '_clip_free'):
                    chat_handler._clip_free()
                
                # Delete the handler reference
                del self.chat_handler_ref
                self.chat_handler_ref = None
                cstr("[SmartLM] ✓ Chat handler freed").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] Warning: Error freeing chat_handler: {e}").warning.print()
        
        if self.model is not None:
            # Close the llama model properly
            if hasattr(self.model, 'close'):
                try:
                    self.model.close()
                except:
                    pass
            
            # Clear model reference
            if hasattr(self.model, 'chat_handler'):
                self.model.chat_handler = None
            
            del self.model
            self.model = None
        if self.processor is not None:
            self.processor = None
        if self.tokenizer is not None:
            self.tokenizer = None
        
        # Clear references and free memory
        self.current_template = None
        self.model_type = ModelType.UNKNOWN
        
        # Force garbage collection
        gc.collect()
        
        # Use ComfyUI's soft_empty_cache for better memory management
        if torch.cuda.is_available():
            try:
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except:
                # Fallback to direct CUDA cache clear
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    @staticmethod
    def cleanup_memory_before_load():
        # Gentle memory cleanup before loading LM model.
        #
        # Unlike Smart Loader (which loads main generation models), Smart LML is typically
        # used mid-workflow after generation models are loaded. We only clear unused cache,
        # not aggressively unload anything that might be needed for generation.
        # Just garbage collect and clear unused cache
        gc.collect()
        
        # Clear unused CUDA cache (doesn't unload models)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use ComfyUI's soft cleanup (safer than aggressive cleanup)
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except:
            pass
    
    @staticmethod
    def tensor_to_pil(tensor) -> Optional[Image.Image]:
        # Convert ComfyUI image tensor to PIL Image
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def get_device_info(self):
        # Get comprehensive device and memory information
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}
        device_type = "cpu"
        recommended_device = "cpu"
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            gpu_info = {
                "available": True,
                "total_memory": total,
                "free_memory": total - (torch.cuda.memory_allocated(0) / (1024**3)),
            }
            device_type = "cuda"
            recommended_device = "cuda"
        elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
            device_type = "mps"
            recommended_device = "mps"
            gpu_info = {"available": True, "total_memory": 0, "free_memory": 0}
        
        sys_mem = psutil.virtual_memory()
        return {
            "gpu": gpu_info,
            "system_memory": {
                "total": sys_mem.total / (1024**3),
                "available": sys_mem.available / (1024**3),
            },
            "device_type": device_type,
            "recommended_device": recommended_device,
            "device": recommended_device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    
    def _get_model_size_from_disk(self, local_path: str, repo_id: str) -> float:
        # Get actual model size from disk if available.
        #
        # Args:
        #     local_path: Local model folder name
        #     repo_id: HuggingFace repo ID
        #
        # Returns:
        #     Model size in GB, or 0 if not found
        try:
            import folder_paths
            from pathlib import Path
            
            # Check local LLM folder
            llm_dir = Path(folder_paths.models_dir) / "LLM"
            model_path = llm_dir / local_path if local_path else None
            
            if model_path and model_path.exists():
                # Sum up all model file sizes
                total_size = 0
                for file in model_path.rglob("*"):
                    if file.is_file() and any(file.name.endswith(ext) for ext in [
                        ".safetensors", ".bin", ".pth", ".gguf", ".pt"
                    ]):
                        total_size += file.stat().st_size
                
                if total_size > 0:
                    return total_size / (1024**3)  # Convert to GB
        except Exception as e:
            # Silently fail, will use fallback estimation
            pass
        
        return 0.0
    
    def enforce_quantization(self, template_name: str, quantization: str, device_info: dict) -> str:
        # Auto-adjust quantization based on available memory.
        #
        # Estimates model memory requirements and downgrades quantization if needed.
        # Rough estimates based on model parameters:
        # - FP16/BF16: ~2 bytes per parameter
        # - 8-bit: ~1 byte per parameter
        # - 4-bit: ~0.5 bytes per parameter
        #
        # Args:
        #     template_name: Model template name
        #     quantization: Requested quantization (auto/fp16/bf16/fp32/8bit/4bit)
        #     device_info: Device memory information
        #
        # Returns:
        #     Adjusted quantization mode
        # If not auto, return as-is
        if quantization != "auto":
            return quantization
        
        template_info = load_template(template_name)
        
        # Get available memory
        if device_info["recommended_device"] in {"cpu", "mps"}:
            available = device_info["system_memory"]["available"]
            cstr(f"[SmartLM Auto] Available system memory: {available:.1f} GB").msg.print()
        else:
            available = device_info["gpu"]["free_memory"]
            cstr(f"[SmartLM Auto] Available GPU memory: {available:.1f} GB").msg.print()
        
        # Check if template has explicit VRAM requirements (preferred, most accurate)
        vram_requirements = template_info.get("vram_requirement", {})
        if vram_requirements:
            # Use values from template and add 20% safety margin (like QwenVL does)
            needed_fp16 = vram_requirements.get("full", 0) * 1.2 if vram_requirements.get("full", 0) > 0 else 0
            needed_8bit = vram_requirements.get("8bit", 0) * 1.2 if vram_requirements.get("8bit", 0) > 0 else 0
            needed_4bit = vram_requirements.get("4bit", 0) * 1.2 if vram_requirements.get("4bit", 0) > 0 else 0
            
            cstr(f"[SmartLM Auto] Using template vram_requirement with 20% safety margin (fp16={needed_fp16:.1f}, 8bit={needed_8bit:.1f}, 4bit={needed_4bit:.1f} GB)").msg.print()
        else:
            # Fallback: estimate from actual disk size or model name
            local_path = template_info.get("local_path", "")
            repo_id = template_info.get("repo_id", "")
            
            actual_size = self._get_model_size_from_disk(local_path, repo_id)
            if actual_size > 0:
                estimated_size = actual_size
                cstr(f"[SmartLM Auto] Detected model size on disk: {estimated_size:.1f} GB").msg.print()
            else:
                # Last resort: estimate from model name
                model_name = repo_id.split("/")[-1].lower() if repo_id else ""
                
                # Known Florence-2 models
                estimated_size = smartlm_florence2.get_florence2_size_estimate(model_name)
                if estimated_size == 0:
                    # QwenVL and other models by parameter count
                    if "2b" in model_name or "2.5b" in model_name:
                        estimated_size = 4.0
                    elif "3b" in model_name:
                        estimated_size = 6.0
                    elif "4b" in model_name:
                        estimated_size = 8.0
                    elif "7b" in model_name or "8b" in model_name:
                        estimated_size = 14.0
                    elif "32b" in model_name:
                        estimated_size = 64.0
                    else:
                        # Unknown size, be conservative
                        estimated_size = 3.0
                        cstr(f"[SmartLM Auto] Unknown model size, estimating {estimated_size:.1f} GB").msg.print()
            
            # Calculate requirements from estimated size (with 20% overhead)
            needed_fp16 = estimated_size * 1.2
            needed_8bit = estimated_size * 0.6
            needed_4bit = estimated_size * 0.4
        
        # Choose quantization based on available memory
        if needed_fp16 < available:
            selected = "fp16"
            cstr(f"[SmartLM Auto] Selected FP16 (needs ~{needed_fp16:.1f} GB)").msg.print()
        elif needed_8bit < available:
            selected = "8bit"
            cstr(f"[SmartLM Auto] Selected 8-bit (needs ~{needed_8bit:.1f} GB)").msg.print()
        elif needed_4bit < available:
            selected = "4bit"
            cstr(f"[SmartLM Auto] Selected 4-bit (needs ~{needed_4bit:.1f} GB)").msg.print()
        else:
            # Not enough memory even for 4-bit
            selected = "4bit"
            cstr(f"[SmartLM Auto] Warning: May not have enough memory. Using 4-bit (needs ~{needed_4bit:.1f} GB, available {available:.1f} GB)").warning.print()
        
        return selected
    
    def ensure_model_path(self, template_name: str) -> tuple[str, str, str]:
        # Download model if needed and return (model_path, model_folder_path, repo_id)
        # Handles legacy templates by searching LLM folder and auto-fixing paths
        template_info = load_template(template_name)
        if not template_info:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Check for local_path first
        local_path = template_info.get("local_path")
        repo_id = template_info.get("repo_id")
        
        # Support repo_id as URL (detect if it's a direct download URL)
        is_direct_url = repo_id and (repo_id.startswith("http://") or repo_id.startswith("https://"))
        
        if not repo_id and not local_path:
            raise ValueError(f"Template '{template_name}' missing repo_id or local_path")
        
        # Determine model directory based on type
        model_type = detect_model_type(template_info)
        
        import folder_paths
        models_base = Path(folder_paths.models_dir) / "LLM"
        
        # Construct target path based on local_path or repo_id
        target = None
        needs_search = False
        
        if local_path:
            # Try to construct path from local_path
            if local_path.lower().endswith(".gguf"):
                # Check if local_path is just filename (legacy) or full path
                if '/' in local_path or '\\' in local_path:
                    # Already has path components - use as-is
                    target = models_base / local_path
                else:
                    # Just filename - construct expected folder structure
                    model_name = Path(local_path).stem
                    if model_type == ModelType.QWENVL or "qwen" in local_path.lower():
                        target = models_base / "Qwen-VL" / model_name / Path(local_path).name
                    else:
                        target = models_base / model_name / Path(local_path).name
            else:
                # Non-GGUF: use specified local path as-is
                target = models_base / local_path
            
            # If constructed target doesn't exist, search for the file (ONLY for GGUF files)
            # Do NOT search for transformers models as multiple models use the same filename (model.safetensors)
            if not target.exists():
                filename = Path(local_path).name  # Extract just the filename
                
                # Only search for unique filenames (GGUF models have unique names)
                # Skip search for generic names like model.safetensors, pytorch_model.bin
                generic_names = {'model.safetensors', 'pytorch_model.bin', 'model.bin', 'adapter_model.bin'}
                
                if filename.lower() not in generic_names and filename.lower().endswith('.gguf'):
                    cstr(f"[SmartLM] Searching for existing GGUF file: {filename}...").msg.print()
                    found_path = search_model_file(filename, models_base)
                    if found_path:
                        target = found_path
                        cstr(f"[SmartLM] ✓ Found existing file at {target}").msg.print()
                # If generic name or not GGUF, don't search - will download to correct location below
        else:
            # No local_path - construct from repo_id
            model_name = repo_id.split("/")[-1]
            
            # For direct GGUF URL downloads, create folder structure properly
            if is_direct_url and model_name.lower().endswith(".gguf"):
                # Extract filename and create model-specific folder
                filename = model_name
                folder_name = Path(filename).stem  # Remove .gguf extension for folder name
                if model_type == ModelType.QWENVL or "qwen" in filename.lower():
                    target = models_base / "Qwen-VL" / folder_name / filename
                else:
                    target = models_base / folder_name / filename
                
                # Search for existing GGUF file if target doesn't exist
                if not target.exists() and filename.lower().endswith('.gguf'):
                    cstr(f"[SmartLM] Searching for existing GGUF file: {filename}...").msg.print()
                    found_path = search_model_file(filename, models_base)
                    if found_path:
                        target = found_path
                        cstr(f"[SmartLM] ✓ Found existing file at {target}").msg.print()
            elif model_type == ModelType.QWENVL:
                target = models_base / "Qwen-VL" / model_name
            else:
                target = models_base / model_name
        
        # Download if not exists
        downloaded = False
        if not target.exists():
            # For transformers models, check if this repo_id is already downloaded elsewhere
            if repo_id and not is_direct_url and not target.name.lower().endswith('.gguf') and models_base.exists():
                # Search for existing model with same repo_id
                for existing_dir in models_base.iterdir():
                    if existing_dir.is_dir():
                        # Check if this directory contains the same model by looking for matching files
                        model_files = list(existing_dir.glob("*.safetensors")) or list(existing_dir.glob("pytorch_model*.bin"))
                        if model_files:
                            # Check if config.json exists and has matching _name_or_path
                            config_file = existing_dir / "config.json"
                            if config_file.exists():
                                try:
                                    import json
                                    config = json.loads(config_file.read_text())
                                    # Check if this is the EXACT same model by comparing repo path
                                    config_model_name = config.get("_name_or_path", "")
                                    # Match if _name_or_path matches repo_id (handles user/repo format)
                                    # Also handle case where config has full path vs just repo name
                                    repo_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
                                    config_name = config_model_name.split('/')[-1] if '/' in config_model_name else config_model_name
                                    
                                    if config_model_name == repo_id or (repo_name and config_name == repo_name):
                                        cstr(f"[SmartLM] ✓ Found existing {repo_id} model at {existing_dir}").msg.print()
                                        target = existing_dir
                                        # Update template with correct local_path
                                        relative_path = (target.relative_to(models_base).as_posix() + '/')
                                        if local_path != relative_path:
                                            update_template_local_path(template_name, relative_path)
                                        break
                                except:
                                    pass
            
            if not target.exists() and is_direct_url:
                # Single file download from direct URL
                cstr(f"[SmartLM] Downloading single file from {repo_id}").msg.print()
                target.parent.mkdir(parents=True, exist_ok=True)
                from huggingface_hub import hf_hub_download
                
                # Parse HuggingFace URL: https://huggingface.co/USER/REPO/resolve/main/file.gguf
                parts = repo_id.split('/')
                if 'huggingface.co' in repo_id and len(parts) >= 6:
                    filename = parts[-1]  # file.gguf
                    
                    # Download directly to target with progress
                    download_with_progress(repo_id, str(target), filename)
                    cstr(f"[SmartLM] ✓ Downloaded to {target}").msg.print()
                    downloaded = True
                else:
                    raise ValueError(f"Invalid repo_id URL format: {repo_id}")
            elif repo_id:
                # Full repository download
                cstr(f"[SmartLM] Downloading {repo_id} to {target}").msg.print()
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", ".git*"],
                )
                downloaded = True
            else:
                raise ValueError(f"Template '{template_name}' local_path '{local_path}' not found at {target}")
        
        # Verify model integrity after download
        if downloaded and target.exists():
            if not verify_model_integrity(target, extract_repo_id_from_url(repo_id)):
                raise RuntimeError(f"Model verification failed for {target}. Please delete and re-download.")
        
        # Update template with local_path after download/verification for offline usage
        # Always calculate relative path and update if different from template
        # Use forward slashes for cross-platform compatibility (JSON standard)
        relative_path = target.relative_to(models_base)
        current_local_path = relative_path.as_posix()
        
        # Add trailing slash for directories (transformers models)
        if target.is_dir() and not current_local_path.endswith('/'):
            current_local_path += '/'
        
        # Update if local_path is empty or different from actual path
        if not local_path or local_path != current_local_path:
            update_template_local_path(template_name, current_local_path)
        
        # Update vram_requirement with actual file size (after download or if file exists)
        if downloaded or target.exists():
            try:
                total_size_gb = calculate_model_size(target)
                
                # For Qwen-VL models, add mmproj file size if it exists
                mmproj_path = template_info.get("mmproj_path")
                if mmproj_path:
                    # Construct mmproj path to check if it exists
                    if '/' in mmproj_path or '\\' in mmproj_path:
                        import folder_paths
                        llm_dir = Path(folder_paths.models_dir) / "LLM"
                        mmproj_file = llm_dir / mmproj_path
                    else:
                        mmproj_file = target.parent / mmproj_path
                    
                    if mmproj_file.exists():
                        mmproj_size_gb = calculate_model_size(mmproj_file)
                        total_size_gb += mmproj_size_gb
                
                if total_size_gb > 0.1:  # Only update if we found significant files (>100MB)
                    vram_full = round(total_size_gb, 1)
                    
                    # Check if model is pre-quantized
                    model_name_lower = str(target).lower()
                    has_quant_markers = any(marker in model_name_lower for marker in [
                        "fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_",
                        "gptq", "awq", "gguf"
                    ])
                    is_gguf = str(target).lower().endswith(".gguf")
                    is_quantized = template_info.get("quantized", has_quant_markers or is_gguf)
                    
                    # Build vram_requirement
                    vram_req = {"full": vram_full}
                    if not is_quantized:
                        vram_req["8bit"] = round(vram_full * 0.5, 1)
                        vram_req["4bit"] = round(vram_full * 0.25, 1)
                    
                    # Update template with actual size
                    if update_template_vram_requirement(template_name, vram_req):
                        cstr(f"[SmartLM] Updated template with actual model size: {vram_full} GB").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] Could not calculate model size after download: {e}").warning.print()
        
        # Return tuple: (model_path, model_folder, repo_id)
        return (str(target), str(target.parent), repo_id or "")
    
    def ensure_mmproj_path(self, template_info: dict, model_folder: str, template_name: str = None) -> Optional[str]:
        # Download mmproj file if needed and return local path. Downloads into model folder.
        mmproj_path = template_info.get("mmproj_path")
        mmproj_url = template_info.get("mmproj_url")
        
        if not mmproj_path and not mmproj_url:
            return None
        
        model_folder_path = Path(model_folder)
        
        # Extract original filename from URL first (needed for both cases)
        original_filename = None
        if mmproj_url:
            original_filename = mmproj_url.split('/')[-1]
        
        # Prioritize generating name from URL to include precision info
        if mmproj_url:
            # Preserve precision info (fp16, bf16, f16, etc.) when renaming
            # Extract precision marker from original filename
            import re
            precision_match = re.search(r'(fp16|bf16|f16|f32)', original_filename.lower())
            precision_suffix = f"-{precision_match.group(1)}" if precision_match else ""
            
            # Get model base name from folder
            model_base = model_folder_path.name
            
            # Construct renamed filename with precision info preserved
            target_filename = f"{model_base}{precision_suffix}.mmproj.gguf"
            target = model_folder_path / target_filename
        elif mmproj_path:
            # Fallback: use mmproj_path if no URL available (offline mode)
            # Check if mmproj_path contains subdirectories (local mmproj from different folder)
            if '/' in mmproj_path or '\\' in mmproj_path:
                # Full relative path from LLM folder (e.g., "Qwen-VL/Model/file.mmproj.gguf")
                import folder_paths
                llm_dir = Path(folder_paths.models_dir) / "LLM"
                target = llm_dir / mmproj_path
            else:
                # Just a filename - should be in the model's folder
                target = model_folder_path / mmproj_path
            target_filename = target.name
        else:
            return None
        
        # If it already exists, return it
        if target.exists():
            return str(target)
        
        # Download if mmproj_url is provided
        if not mmproj_url:
            cstr(f"[SmartLM] Warning: mmproj_path specified but file not found and no mmproj_url to download: {target}").warning.print()
            return None
        
        # Download if not exists
        if not target.exists() and mmproj_url:
            cstr(f"[SmartLM] Downloading MMProj from {mmproj_url}").msg.print()
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse HuggingFace URL
            parts = mmproj_url.split('/')
            if 'huggingface.co' in mmproj_url and len(parts) >= 6:
                # Download directly to target with renamed filename
                download_with_progress(mmproj_url, str(target), target_filename)
                cstr(f"[SmartLM] ✓ MMProj downloaded as {target_filename}").msg.print()
                
                # Verify mmproj file integrity after download using original filename
                if target.exists():
                    # Pass original filename from URL so HuggingFace can find the hash
                    if not verify_model_integrity(target, extract_repo_id_from_url(mmproj_url), original_filename):
                        cstr(f"[SmartLM] Warning: MMProj verification failed for {target_filename}").warning.print()
                        # Don't raise error - allow usage but warn user
                    
                    # Update template with the new mmproj filename (just the filename, not full path)
                    if template_name:
                        update_template_mmproj_path(template_name, target_filename)
            else:
                cstr(f"[SmartLM] Warning: Invalid mmproj_url format: {mmproj_url}").warning.print()
                return None
        
        return str(target) if target.exists() else None
    
    def load_model(self, template_name: str, quantization: str = "fp16", 
                   attention: str = "sdpa", device: str = "cuda", context_size: int = 32768,
                   memory_cleanup: bool = True, use_torch_compile: bool = False):
        # Load Smart LML model based on template type
        # Proactive memory cleanup before loading (like Smart Loader)
        if memory_cleanup:
            self.cleanup_memory_before_load()
        
        # Auto-detect quantization if set to "auto"
        device_info = self.get_device_info()
        quantization = self.enforce_quantization(template_name, quantization, device_info)
        
        # Check if we need to force cleanup when keep_model_loaded is active
        needs_cleanup = False
        cleanup_reason = []
        
        if self.model is not None:
            # Check if template is changing (different model)
            template_changed = hasattr(self, 'current_template') and self.current_template != template_name
            if template_changed:
                needs_cleanup = True
                cleanup_reason.append(f"template changed ({self.current_template} -> {template_name})")
            
            # Check if quantization mode is changing (critical for VRAM/device_map conflicts)
            is_quantized_new = quantization in ["4bit", "8bit"]
            is_quantized_old = hasattr(self, 'is_quantized') and self.is_quantized
            quantization_mode_changed = is_quantized_new != is_quantized_old
            if quantization_mode_changed:
                needs_cleanup = True
                cleanup_reason.append(f"quantization mode changed ({is_quantized_old} -> {is_quantized_new})")
        
        # Force cleanup if needed
        if needs_cleanup:
            cstr(f"[SmartLM] Forcing cleanup and reload: {', '.join(cleanup_reason)}").msg.print()
            self.clear()
        
        template_info = load_template(template_name)
        model_type = detect_model_type(template_info)
        
        if model_type == ModelType.QWENVL:
            smartlm_qwenvl.load_qwenvl(self, template_name, quantization, attention, device, context_size, use_torch_compile)
        elif model_type == ModelType.FLORENCE2:
            smartlm_florence2.load_florence2_model(self, template_name, quantization, attention, device, use_torch_compile)
        elif model_type == ModelType.LLM:
            smartlm_llm.load_llm(self, template_name, quantization, attention, device, context_size)
        else:
            raise ValueError(f"Unknown model type for template '{template_name}'")
        
        self.current_template = template_name
        self.model_type = model_type
    
    def generate(self, image: Any, prompt: str, task: Optional[str] = None,
                 text_input: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, num_beams: int = 3, do_sample: bool = True, seed: Optional[int] = None,
                 repetition_penalty: float = 1.0, frame_count: int = 8,
                 llm_mode: str = "direct_chat", instruction_template: str = "",
                 convert_to_bboxes: bool = True, detection_filter_threshold: float = 0.80,
                 nms_iou_threshold: float = 0.50) -> tuple[str, dict]:
        # Unified generation method for all model types - returns (text, data_dict)
        if self.model_type == ModelType.QWENVL:
            text = smartlm_qwenvl.generate_qwenvl(self, image, prompt, max_tokens, temperature, top_p, top_k, num_beams, do_sample, seed, repetition_penalty, frame_count)
            return (text, {})
        elif self.model_type == ModelType.FLORENCE2:
            return smartlm_florence2.generate_florence2(self, image, task or prompt, max_tokens, num_beams, do_sample, seed, repetition_penalty, text_input, convert_to_bboxes, detection_filter_threshold, nms_iou_threshold)
        elif self.model_type == ModelType.LLM:
            text = smartlm_llm.generate_llm(self, prompt, max_tokens, temperature, top_p, top_k, seed, repetition_penalty, llm_mode, instruction_template)
            return (text, {})
        else:
            raise ValueError(f"No model loaded or unknown model type: {self.model_type}")