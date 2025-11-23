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
from pathlib import Path
from typing import Dict, Optional, Any, List
from PIL import Image
from enum import Enum

# Import cstr for logging
from . import cstr

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

# Use Eclipse folder if available, otherwise fallback to repo
TEMPLATE_DIR = ECLIPSE_TEMPLATE_DIR if ECLIPSE_TEMPLATE_DIR.exists() else REPO_TEMPLATE_DIR

# Config files: check Eclipse first, fallback to repo
ECLIPSE_CONFIG_DIR = Path(folder_paths.models_dir) / "Eclipse" / "config"
REPO_CONFIG_DIR = NODE_DIR / "templates" / "config"

PROMPT_CONFIG_PATH = (ECLIPSE_CONFIG_DIR / "smartlm_prompt_defaults.json" 
                      if (ECLIPSE_CONFIG_DIR / "smartlm_prompt_defaults.json").exists() 
                      else REPO_CONFIG_DIR / "smartlm_prompt_defaults.json")
LLM_FEW_SHOT_CONFIG_PATH = (ECLIPSE_CONFIG_DIR / "llm_few_shot_training.json" 
                            if (ECLIPSE_CONFIG_DIR / "llm_few_shot_training.json").exists() 
                            else REPO_CONFIG_DIR / "llm_few_shot_training.json")

def download_with_progress(url: str, path: str, name: str) -> None:
    # Download file with progress bar
    import urllib.request
    from tqdm import tqdm
    
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc=f'[SmartLM] Downloading {name}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

# Ensure template directory exists
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# Global configuration storage
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}
FLORENCE_TASKS = {}
LLM_FEW_SHOT_EXAMPLES = {}  # LLM few-shot training examples
LLAMA_CPP_AVAILABLE = False  # Will be set during initialization
LLAMA_CPP_MODULE = None  # Will be "llama_cpp_cuda" or "llama_cpp"

def get_template_list() -> List[str]:
    # Get list of available Smart LML templates
    templates = []
    if TEMPLATE_DIR.exists():
        for f in TEMPLATE_DIR.iterdir():
            if f.suffix == '.json' and not f.name.startswith('_'):
                templates.append(f.stem)
    return sorted(templates) if templates else ["Qwen2.5-VL-3B-Instruct"]

def get_default_template() -> str:
    # Get the template marked as default, or first available
    if TEMPLATE_DIR.exists():
        for f in TEMPLATE_DIR.iterdir():
            if f.suffix == '.json' and not f.name.startswith('_'):
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        if data.get('default', False):
                            return f.stem
                except Exception:
                    continue
    # Fallback to first template or hardcoded default
    templates = get_template_list()
    return templates[0] if templates else "Qwen2.5-VL-3B-Instruct"

def load_template(name: str) -> dict:
    # Load a Smart LML template configuration
    if not name or name == "None":
        return {}
    template_path = TEMPLATE_DIR / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        cstr(f"[SmartLM] Error loading template {name}: {e}").error.print()
    return {}

def update_template_local_path(name: str, local_path: str) -> bool:
    # Update template's local_path field after successful download for offline usage
    if not name or name == "None":
        return False
    template_path = TEMPLATE_DIR / f"{name}.json"
    try:
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Only update if local_path is empty or different
            if template_data.get("local_path") != local_path:
                template_data["local_path"] = local_path
                with open(template_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
                cstr(f"[SmartLM] ✓ Updated template '{name}' with local_path: {local_path}").msg.print()
                return True
    except Exception as e:
        cstr(f"[SmartLM] Warning: Could not update template {name}: {e}").warning.print()
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
    
    template_path = TEMPLATE_DIR / f"{name}.json"
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

def get_llm_model_list() -> list:
    # Scan models/LLM folder and return list of available models.
    # Returns both folders (model repos) and individual files (.safetensors, .gguf, etc.)
    # Scans recursively to find nested model folders (e.g., /LLM/qwen-vl/Qwen2-VL-2B-Instruct/)
    try:
        import folder_paths
        llm_dir = Path(folder_paths.models_dir) / "LLM"
        
        if not llm_dir.exists():
            return ["(No models/LLM folder found)"]
        
        models = []
        model_folders_with_shards = set()  # Track folders with shard files to show folder instead of files
        model_extensions = ['.safetensors', '.gguf', '.bin', '.pt']
        
        def scan_for_models(base_path: Path, relative_path: str = ""):
            """Recursively scan for model folders and files"""
            try:
                for item in base_path.iterdir():
                    if item.is_dir():
                        # Build relative path
                        item_rel_path = f"{relative_path}/{item.name}" if relative_path else item.name
                        
                        # Check if it's a valid model folder (contains config.json or model files)
                        has_config = (item / "config.json").exists()
                        model_files = [
                            f for f in item.iterdir()
                            if f.is_file() and f.suffix in model_extensions
                        ]
                        
                        # Check if folder has shard files
                        has_shards = any('-of-' in f.name or '.shard' in f.name.lower() for f in model_files)
                        
                        if has_config or model_files:
                            # If folder has shards or config.json, show folder instead of individual files
                            if has_shards or has_config:
                                models.append(item_rel_path + "/")
                                model_folders_with_shards.add(item_rel_path)
                            # If single file in folder (like single GGUF), add the file path instead
                            elif len(model_files) == 1:
                                single_file = model_files[0]
                                models.append(f"{item_rel_path}/{single_file.name}")
                                model_folders_with_shards.add(item_rel_path)  # Mark to skip later
                            # Multiple non-shard files - add folder
                            else:
                                models.append(item_rel_path + "/")
                                model_folders_with_shards.add(item_rel_path)
                        
                        # Recurse into subdirectories (limit depth to avoid infinite loops)
                        if relative_path.count('/') < 3:  # Max 3 levels deep
                            scan_for_models(item, item_rel_path)
                    
                    elif item.is_file() and item.suffix in model_extensions:
                        # Individual model file - only add if not inside a model folder already processed
                        # Skip shard files (model-00001-of-00002.safetensors, etc.)
                        is_shard = '-of-' in item.name or '.shard' in item.name.lower()
                        
                        # Check if the parent directory of this file is already processed
                        parent_already_added = relative_path in model_folders_with_shards
                        
                        if not is_shard and not parent_already_added:
                            if relative_path:
                                models.append(f"{relative_path}/{item.name}")
                            else:
                                models.append(item.name)
            except PermissionError:
                pass  # Skip directories we can't access
        
        # Start recursive scan from LLM root
        scan_for_models(llm_dir)
        
        if not models:
            return ["(No models found in models/LLM)"]
        
        return sorted(models)
    
    except Exception as e:
        cstr(f"[SmartLM] Error scanning models/LLM: {e}").error.print()
        return ["(Error scanning models folder)"]

def get_mmproj_list() -> list:
    # Scan models/LLM folder for mmproj files for GGUF QwenVL models.
    # Returns .mmproj files and .gguf files containing 'mmproj' in the name.
    # Scans recursively to find nested mmproj files.
    try:
        import folder_paths
        llm_dir = Path(folder_paths.models_dir) / "LLM"
        
        if not llm_dir.exists():
            return ["None", "(No models/LLM folder found)"]
        
        mmproj_files = ["None"]  # Add None option for when mmproj is not needed
        
        def scan_for_mmproj(base_path: Path, relative_path: str = ""):
            """Recursively scan for mmproj files"""
            try:
                for item in base_path.iterdir():
                    if item.is_file():
                        # Match .mmproj files or .gguf files with 'mmproj' in name
                        if item.suffix == '.mmproj' or (item.suffix == '.gguf' and 'mmproj' in item.name.lower()):
                            if relative_path:
                                mmproj_files.append(f"{relative_path}/{item.name}")
                            else:
                                mmproj_files.append(item.name)
                    
                    elif item.is_dir():
                        # Recurse into subdirectories (limit depth to avoid infinite loops)
                        item_rel_path = f"{relative_path}/{item.name}" if relative_path else item.name
                        if relative_path.count('/') < 3:  # Max 3 levels deep
                            scan_for_mmproj(item, item_rel_path)
            except PermissionError:
                pass  # Skip directories we can't access
        
        # Start recursive scan from LLM root
        scan_for_mmproj(llm_dir)
        
        if len(mmproj_files) == 1:  # Only "None" option
            mmproj_files.append("(No mmproj files found)")
        
        return sorted(mmproj_files)
    
    except Exception as e:
        cstr(f"[SmartLM] Error scanning for mmproj files: {e}").error.print()
        return ["None", "(Error scanning mmproj files)"]

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
    global MODEL_CONFIGS, SYSTEM_PROMPTS, FLORENCE_TASKS, LLM_FEW_SHOT_EXAMPLES
    
    try:
        with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
            # QwenVL prompts
            SYSTEM_PROMPTS = config_data.get("_system_prompts", {})
            MODEL_CONFIGS["_preset_prompts"] = config_data.get("_preset_prompts", [])
            
            # Florence-2 tasks
            FLORENCE_TASKS = config_data.get("_florence2_tasks", {
                "caption": "<CAPTION>",
                "detailed_caption": "<DETAILED_CAPTION>",
                "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
                "prompt_gen_tags": "<GENERATE_TAGS>",
                "prompt_gen_mixed_caption": "<MIXED_CAPTION>",
                "prompt_gen_analyze": "<ANALYZE>",
                "prompt_gen_mixed_caption_plus": "<MIXED_CAPTION_PLUS>",
            })
            
        cstr(f"[SmartLM] Loaded {len(SYSTEM_PROMPTS)} QwenVL prompts, {len(FLORENCE_TASKS)} Florence-2 tasks").msg.print()
    except Exception as exc:
        cstr(f"[SmartLM] Config load failed: {exc}").error.print()
        SYSTEM_PROMPTS = {}
        MODEL_CONFIGS["_preset_prompts"] = []
        FLORENCE_TASKS = {}
    
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
                if "florence-2-base" in model_name or "florence2-base" in model_name:
                    estimated_size = 0.5  # ~230M params
                elif "florence-2-large" in model_name or "florence2-large" in model_name:
                    estimated_size = 1.5  # ~770M params
                # QwenVL and other models by parameter count
                elif "2b" in model_name or "2.5b" in model_name:
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
    
    def ensure_model_path(self, template_name: str) -> tuple[str, str]:
        # Download model if needed and return (model_path, model_folder_path)
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
        
        if local_path:
            # For GGUF files, create model-specific folder
            if local_path.lower().endswith(".gguf"):
                # Extract filename without extension for folder name
                model_name = Path(local_path).stem
                # For Qwen models, use Qwen-VL subfolder for compatibility with other nodes
                if model_type == ModelType.QWENVL or "qwen" in local_path.lower():
                    target = models_base / "Qwen-VL" / model_name / Path(local_path).name
                else:
                    target = models_base / model_name / Path(local_path).name
            else:
                # Non-GGUF: use specified local path as-is
                target = models_base / local_path
        else:
            # Use repo name - for QwenVL use Qwen-VL subfolder, for Florence-2 use repo name
            model_name = repo_id.split("/")[-1]
            if model_type == ModelType.QWENVL:
                target = models_base / "Qwen-VL" / model_name
            else:
                target = models_base / model_name
        
        # Download if not exists
        downloaded = False
        if not target.exists():
            if is_direct_url:
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
        
        # Update template with local_path after successful download for offline usage
        if downloaded and not local_path:
            # Calculate relative path from models/LLM
            relative_path = target.relative_to(models_base)
            update_template_local_path(template_name, str(relative_path))
        
        # Return both the model file path and its parent folder
        return (str(target), str(target.parent))
    
    def ensure_mmproj_path(self, template_info: dict, model_folder: str) -> Optional[str]:
        # Download mmproj file if needed and return local path. Downloads into model folder.
        mmproj_path = template_info.get("mmproj_path")
        mmproj_url = template_info.get("mmproj_url")
        
        if not mmproj_path and not mmproj_url:
            return None
        
        model_folder_path = Path(model_folder)
        
        # Use mmproj_path as the target filename (already includes renamed filename in template)
        if mmproj_path:
            target_filename = mmproj_path
        elif mmproj_url:
            # Fallback: use original filename from URL if no mmproj_path specified
            target_filename = mmproj_url.split('/')[-1]
        else:
            return None
        
        # Target location: inside the model's folder
        target = model_folder_path / target_filename
        
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
            self._load_qwenvl(template_name, quantization, attention, device, context_size, use_torch_compile)
        elif model_type == ModelType.FLORENCE2:
            self._load_florence2(template_name, quantization, attention, device, use_torch_compile)
        elif model_type == ModelType.LLM:
            self._load_llm(template_name, quantization, attention, device, context_size)
        else:
            raise ValueError(f"Unknown model type for template '{template_name}'")
        
        self.current_template = template_name
        self.model_type = model_type
    
    def _load_qwenvl(self, template_name: str, quantization: str, attention: str, device: str, context_size: int = 32768, use_torch_compile: bool = False):
        # Load QwenVL model (supports both transformers and GGUF)
        template_info = load_template(template_name)
        model_path, model_folder = self.ensure_model_path(template_name)
        
        # Check if this is a GGUF model
        local_path = template_info.get("local_path", "")
        is_gguf = local_path.lower().endswith(".gguf")
        
        if is_gguf:
            # Load GGUF model with llama-cpp-python
            self._load_qwenvl_gguf(template_name, template_info, device, context_size)
            return
        
        # Load transformers model
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        from transformers import BitsAndBytesConfig
        
        # Auto-detect if model is pre-quantized (FP8 models usually have it in name/folder)
        repo_id = template_info.get("repo_id", "")
        has_quant_markers = any(marker in local_path.lower() or marker in repo_id.lower() or marker in model_path.lower()
                                for marker in ["fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_"])
        
        # Auto-detect, but allow template override (GGUF already handled above)
        is_prequantized = template_info.get("quantized", has_quant_markers)
        
        # Determine quantization config and dtype (matching qwenvl_base logic)
        if is_prequantized:
            # Pre-quantized models (FP8) - no additional quantization
            quant_config = None
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            dtype = None  # BitsAndBytes handles dtype
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None  # BitsAndBytes handles dtype
        else:
            # Non-quantized mode: fp16, bf16, or fp32
            quant_config = None
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            dtype = dtype_map.get(quantization, torch.float16)
            # Fallback to fp32 on CPU if not fp32 already
            if device == "cpu" and dtype != torch.float32:
                dtype = torch.float32
        
        # Resolve attention mode (QwenVL doesn't support "auto")
        if attention == "auto" or attention == "flash_attention_2":
            # Try flash_attention_2 if available
            try:
                import flash_attn
                attention = "flash_attention_2"
            except ImportError:
                attention = "sdpa"
                if attention == "flash_attention_2":
                    cstr("[SmartLM] Flash-Attn requested but unavailable, falling back to SDPA").warning.print()
        elif attention == "eager":
            attention = "eager"
        else:
            attention = "sdpa"
        
        # Build load_kwargs based on model type
        load_kwargs = {
            "dtype": dtype,
            "attn_implementation": attention,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,  # Required to avoid "Cannot copy out of meta tensor" errors
        }
        
        # HYBRID APPROACH: quantized vs non-quantized
        if is_prequantized:
            # Pre-quantized models: load without device_map to avoid dispatch issues
            load_kwargs["device_map"] = None
            cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [pre-quantized]").msg.print()
            
        elif quant_config:
            # Quantized (4bit/8bit): MUST use device_map="auto" for BitsAndBytes
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
            cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [quantized, device_map=auto]").msg.print()
            
        else:
            # Non-quantized (fp16/bf16): Use ComfyUI offload for better integration
            load_kwargs["device_map"] = None
            cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [non-quantized, ComfyUI offload]").msg.print()
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        
        # Post-loading device placement
        if is_prequantized and device == "cuda" and torch.cuda.is_available():
            # Pre-quantized: manually move to device
            self.model = self.model.to("cuda")
        elif not quant_config:
            # Non-quantized: use ComfyUI offload device
            try:
                import comfy.model_management as mm
                offload_device = mm.unet_offload_device()
                self.model = self.model.to(offload_device)
            except:
                pass
        
        # Track quantization state for generation
        # Pre-quantized models are quantized, even though quant_config is None
        self.is_quantized = is_prequantized or bool(quant_config)
        
        # Apply torch.compile if requested (requires CUDA and Torch 2.1+)
        if use_torch_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                cstr("[SmartLM] ✓ torch.compile enabled for QwenVL").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] torch.compile skipped: {e}").warning.print()
        
        # Enable KV cache for better performance
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    def _load_qwenvl_gguf(self, template_name: str, template_info: dict, device: str, context_size: int = 32768):
        # Load QwenVL GGUF model with llama-cpp-python
        # Check if llama-cpp-python is available (detected at startup)
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models but was not found at startup. "
                "Install with: pip install llama-cpp-python\n"
                "Then restart ComfyUI."
            )
        
        # Get model path first to detect model type from actual filename
        model_path, model_folder = self.ensure_model_path(template_name)
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"GGUF model file not found: {model_file}")
        
        # Detect model type from actual model filename to choose appropriate chat handler
        model_name_lower = model_file.name.lower()
        is_llava = "llava" in model_name_lower
        is_qwen = "qwen" in model_name_lower
        
        # Import llama.cpp components
        ChatHandler: Optional[Any] = None
        try:
            from llama_cpp import Llama
            
            # Import appropriate chat handler for vision support
            if is_llava:
                try:
                    from llama_cpp.llama_chat_format import Llava16ChatHandler
                    ChatHandler = Llava16ChatHandler
                except ImportError:
                    cstr(f"[SmartLM] Warning: Llava16ChatHandler not available").warning.print()
                    
            elif is_qwen:
                # Try to import Qwen handler (name changed between versions)
                # Note: Qwen2.5-VL support added in llama-cpp-python v0.3.10+
                try:
                    # Try Qwen25VLChatHandler (v0.3.10+, name confirmed in v0.3.16)
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    ChatHandler = Qwen25VLChatHandler
                    cstr(f"[SmartLM] Using Qwen25VLChatHandler").msg.print()
                except ImportError:
                    try:
                        # Try Qwen2VLChatHandler (fallback for potential intermediate versions)
                        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                        ChatHandler = Qwen2VLChatHandler
                        cstr(f"[SmartLM] Using Qwen2VLChatHandler").msg.print()
                    except ImportError:
                        # Handler not available - need v0.3.10 or newer
                        import llama_cpp
                        version = getattr(llama_cpp, '__version__', 'unknown')
                        cstr(f"[SmartLM] Qwen chat handler not available in llama-cpp-python {version}").warning.print()
                        cstr(f"[SmartLM] Qwen2.5-VL requires llama-cpp-python >= 0.3.10").warning.print()
                        cstr(f"[SmartLM] Upgrade with: pip install --upgrade llama-cpp-python>=0.3.16").warning.print()
                        ChatHandler = None
                        
        except ImportError as e:
            raise ImportError(
                f"Failed to import llama-cpp-python components: {e}\n"
                "Install with: pip install llama-cpp-python\n"
                "Then restart ComfyUI."
            )
        
        # MMProj is required for vision support - use helper to download if needed
        mmproj_file_path = self.ensure_mmproj_path(template_info, model_folder)
        mmproj_file = Path(mmproj_file_path) if mmproj_file_path else None
        
        if not mmproj_file:
            cstr(f"[SmartLM] Warning: No MMProj available. Vision features may not work.").warning.print()
        
        cstr(f"[SmartLM] Loading GGUF model: {model_file.name}").msg.print()
        if mmproj_file:
            cstr(f"[SmartLM] Using MMProj: {mmproj_file.name}").msg.print()
        
        # Configure llama.cpp parameters
        n_gpu_layers = -1 if device == "cuda" and torch.cuda.is_available() else 0
        
        # Create chat handler with mmproj if available
        chat_handler = None
        if mmproj_file and ChatHandler is not None:
            try:
                chat_handler = ChatHandler(
                    clip_model_path=str(mmproj_file),
                    verbose=False
                )
                handler_type = "LLaVA 1.6" if is_llava else "Qwen2-VL"
                cstr(f"[SmartLM] ✓ MMProj loaded successfully ({handler_type} handler)").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] Failed to load MMProj: {e}").error.print()
                chat_handler = None
        elif mmproj_file and ChatHandler is None:
            cstr(f"[SmartLM] Warning: MMProj file found but no chat handler available for this model type").warning.print()
        
        # Load GGUF model
        # Note: llama.cpp handles its own quantization and device management
        # We don't use ComfyUI device management or BitsAndBytes for GGUF models
        # Vision models need large context for image embeddings (especially videos)
        # Each frame can use 500-1000 tokens, so 16 frames = 8k-16k tokens minimum
        self.model = Llama(
            model_path=str(model_file),
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,  # llama.cpp manages GPU offloading
            n_ctx=context_size,  # User-configurable context size
            n_batch=512,  # Batch size for processing
            verbose=False,
        )
        
        # CRITICAL: Store chat_handler reference separately for cleanup
        # The chat_handler is created BEFORE Llama() and holds the CLIP model in VRAM
        # We must track it separately to properly free VRAM on cleanup
        self.chat_handler_ref = chat_handler
        
        # Mark as GGUF model
        self.is_gguf = True
        self.is_quantized = True  # GGUF models are pre-quantized (Q8, Q4, etc.)
        
        # Store template for reference
        self.gguf_template = template_info
        
        cstr(f"[SmartLM] ✓ GGUF model loaded with {n_gpu_layers} GPU layers").msg.print()
    
    def _load_llm(self, template_name: str, quantization: str, attention: str, device: str, context_size: int = 32768):
        # Load text-only LLM model (GGUF only)
        template_info = load_template(template_name)
        local_path = template_info.get("local_path", "")
        
        # Only GGUF models supported for text-only LLM
        is_gguf = local_path.lower().endswith(".gguf")
        
        if is_gguf:
            self._load_llm_gguf(template_name, template_info, device, context_size)
        else:
            raise ValueError(f"Text-only LLM models currently only support GGUF format. Got: {local_path}")
    
    def _load_llm_gguf(self, template_name: str, template_info: dict, device: str, context_size: int = 32768):
        # Load text-only GGUF model with llama-cpp-python (no vision support)
        # Check if llama-cpp-python is available
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models but was not found at startup. "
                "Install with: pip install llama-cpp-python\n"
                "Then restart ComfyUI."
            )
        
        # Import llama.cpp
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                f"Failed to import llama-cpp-python: {e}\n"
                "Install with: pip install llama-cpp-python\n"
                "Then restart ComfyUI."
            )
        
        # Get model path - ensure_model_path handles auto-download
        model_path, model_folder = self.ensure_model_path(template_name)
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"GGUF model file not found: {model_file}")
        
        cstr(f"[SmartLM] Loading text-only LLM GGUF model: {model_file.name}").msg.print()
        
        # Configure llama.cpp parameters
        n_gpu_layers = -1 if device == "cuda" and torch.cuda.is_available() else 0
        
        # Load GGUF model (NO chat_handler for text-only)
        self.model = Llama(
            model_path=str(model_file),
            n_gpu_layers=n_gpu_layers,
            n_ctx=context_size,  # User-configurable context size
            verbose=False,
        )
        
        # Mark as GGUF model
        self.is_gguf = True
        self.is_quantized = True
        
        # Store template for reference
        self.gguf_template = template_info
        
        cstr(f"[SmartLM] ✓ Text-only LLM GGUF loaded with {n_gpu_layers} GPU layers").msg.print()
    
    def _load_florence2(self, template_name: str, quantization: str, attention: str, device: str, use_torch_compile: bool = False):
        # Load Florence-2 model using wrapper for proper import handling
        from transformers import BitsAndBytesConfig
        from .florence2_wrapper import load_florence2_model, load_florence2_processor
        
        model_path, model_folder = self.ensure_model_path(template_name)
        
        # Determine quantization config and dtype (same as QwenVL)
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            dtype = None  # BitsAndBytes handles dtype
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None  # BitsAndBytes handles dtype
        else:
            # fp16/bf16/fp32 mode
            quant_config = None
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            dtype = dtype_map.get(quantization, torch.float16)
        
        # Build load_kwargs
        load_kwargs = {
            "dtype": dtype,
            "attn_implementation": attention,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        
        # HYBRID APPROACH: quantized vs non-quantized
        if quant_config:
            # Quantized (4bit/8bit): MUST use device_map="auto" for BitsAndBytes
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
            cstr(f"[SmartLM] Loading Florence-2 {template_name} ({quantization}, attn={attention}) [quantized, device_map=auto]").msg.print()
        else:
            # Non-quantized (fp16/bf16): Use ComfyUI offload for better integration
            load_kwargs["device_map"] = None
            cstr(f"[SmartLM] Loading Florence-2 {template_name} ({quantization}, attn={attention}) [non-quantized, ComfyUI offload]").msg.print()
        
        # Load model using wrapper (handles custom implementation and fallback)
        self.model = load_florence2_model(model_path, **load_kwargs).eval()
        
        # Post-loading device placement for non-quantized models
        if not quant_config:
            try:
                import comfy.model_management as mm
                offload_device = mm.unet_offload_device()
                self.model = self.model.to(offload_device)
            except:
                pass
        
        # Apply torch.compile if requested (requires CUDA and non-quantized model)
        if use_torch_compile and not quant_config and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                cstr("[SmartLM] ✓ torch.compile enabled for Florence-2").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] torch.compile skipped: {e}").warning.print()
        elif use_torch_compile and quant_config:
            cstr("[SmartLM] torch.compile not supported with quantized models").warning.print()
        
        # Load processor using wrapper
        self.processor = load_florence2_processor(model_path)
        self.dtype = dtype if dtype else torch.float16
        
        # Track quantization state for generation
        self.is_quantized = bool(quant_config)
    
    def generate(self, image: Any, prompt: str, task: Optional[str] = None,
                 text_input: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, num_beams: int = 3, do_sample: bool = True, seed: Optional[int] = None,
                 repetition_penalty: float = 1.0, frame_count: int = 8,
                 llm_mode: str = "direct_chat", instruction_template: str = "",
                 convert_to_bboxes: bool = True) -> tuple[str, dict]:
        # Unified generation method for all model types - returns (text, data_dict)
        if self.model_type == ModelType.QWENVL:
            text = self._generate_qwenvl(image, prompt, max_tokens, temperature, top_p, top_k, num_beams, do_sample, seed, repetition_penalty, frame_count)
            return (text, {})
        elif self.model_type == ModelType.FLORENCE2:
            return self._generate_florence2(image, task or prompt, max_tokens, num_beams, do_sample, seed, repetition_penalty, text_input, convert_to_bboxes)
        elif self.model_type == ModelType.LLM:
            text = self._generate_llm(prompt, max_tokens, temperature, top_p, top_k, seed, repetition_penalty, llm_mode, instruction_template)
            return (text, {})
        else:
            raise ValueError(f"No model loaded or unknown model type: {self.model_type}")
    
    def _generate_qwenvl(self, image: Any, prompt: str, max_tokens: int, 
                         temperature: float, top_p: float, top_k: int, num_beams: int, 
                         do_sample: bool, seed: Optional[int], repetition_penalty: float = 1.0, 
                         frame_count: int = 8) -> str:
        # Generate with QwenVL (supports both transformers and GGUF)
        
        # Seed is already clamped by JavaScript for QwenVL (int32 range)
        # Florence-2 and LLM use full uint32 range
        
        # Check if this is a GGUF model
        if hasattr(self, 'is_gguf') and self.is_gguf:
            return self._generate_qwenvl_gguf(image, prompt, max_tokens, temperature, top_p, top_k, seed, repetition_penalty, frame_count)
        
        # Transformers model generation
        import numpy as np
        
        # HYBRID APPROACH: Handle device management based on quantization
        # Set defaults first to ensure variables are always defined
        device = next(self.model.parameters()).device
        offload_device = device
        
        if hasattr(self, 'is_quantized') and not self.is_quantized:
            # Non-quantized: Use ComfyUI device management
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
                offload_device = mm.unet_offload_device()
                self.model.to(device)
            except:
                # ComfyUI not available, use model's current device
                pass
        # Quantized: Model stays where device_map placed it (uses defaults set above)
        
        # Handle video frames if input has multiple frames
        frames = None
        if image is not None and len(image.shape) == 4 and image.shape[0] > 1:
            # This is a video (multiple frames) - limit to frame_count
            total_frames = image.shape[0]
            actual_frame_count = min(frame_count, total_frames)
            frames = [self.tensor_to_pil(image[i]) for i in range(actual_frame_count)]
        
        image_pil = self.tensor_to_pil(image) if image is not None else None
        
        conversation: list[dict[str, Any]] = [{"role": "user", "content": []}]
        
        # Add image if single frame
        if image_pil and frames is None:
            conversation[0]["content"].append({"type": "image", "image": image_pil})
        
        # Add video if multiple frames
        if frames and len(frames) > 1:
            conversation[0]["content"].append({"type": "video", "video": frames})
        
        conversation[0]["content"].append({"type": "text", "text": prompt})
        
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images = [image_pil] if (image_pil and frames is None) else None
        videos = [frames] if frames and len(frames) > 1 else None
        
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        model_inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        
        stop_tokens: list[int] = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": 1.0,  # Default value matching qwenvl_base
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        
        outputs = self.model.generate(**model_inputs, **kwargs)
        
        # Synchronize CUDA to ensure generation is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        
        # HYBRID APPROACH: Offload non-quantized models back
        if hasattr(self, 'is_quantized') and not self.is_quantized and offload_device != device:
            try:
                self.model.to(offload_device)
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except:
                pass
        
        return text.strip()
    
    def _generate_qwenvl_gguf(self, image: Any, prompt: str, max_tokens: int,
                              temperature: float, top_p: float, top_k: int, seed: Optional[int],
                              repetition_penalty: float = 1.0, frame_count: int = 8) -> str:
        # Generate with QwenVL GGUF model using llama-cpp-python
        import base64
        from io import BytesIO
        
        # Set seed if provided (already clamped by JavaScript)
        if seed is not None:
            self.model.set_seed(seed)
        
        # Extract system prompt if embedded in the prompt (formatted as "system\n\nuser_prompt")
        system_prompt = "You are a helpful assistant."
        user_prompt = prompt
        
        if "\n\n" in prompt:
            parts = prompt.split("\n\n", 1)
            if len(parts) == 2:
                # Check if first part looks like a system instruction
                first_part = parts[0].strip()
                if len(first_part) > 10 and not first_part.endswith("?"):
                    system_prompt = first_part
                    user_prompt = parts[1].strip()
        
        # Prepare messages for chat completion with system prompt
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        
        # Add image if provided
        if image is not None:
            # Handle video (multiple frames) or single image
            if len(image.shape) == 4 and image.shape[0] > 1:
                # For video, llama.cpp expects images in the content array - limit to frame_count
                total_frames = image.shape[0]
                actual_frame_count = min(frame_count, total_frames)
                image_content = []
                for i in range(actual_frame_count):
                    pil_image = self.tensor_to_pil(image[i])
                    # Convert to base64 data URL for llama.cpp
                    buffered = BytesIO()
                    try:
                        pil_image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        image_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        })
                    finally:
                        # Cleanup memory per frame
                        buffered.close()
                        del pil_image, buffered
                
                # Force garbage collection after processing all frames
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add text prompt (use extracted user prompt)
                image_content.append({"type": "text", "text": user_prompt})
                
                messages.append({
                    "role": "user",
                    "content": image_content
                })
            else:
                # Single image
                pil_image = self.tensor_to_pil(image)
                buffered = BytesIO()
                try:
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                            {"type": "text", "text": user_prompt}
                        ]
                    })
                finally:
                    # Cleanup memory
                    buffered.close()
                    del pil_image, buffered
        else:
            # Text-only prompt (use extracted user prompt)
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        # Generate response
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stream=False
            )
            
            # Extract text from response
            text = response['choices'][0]['message']['content']
            
            # Clear messages to free base64 image data from memory
            messages.clear()
            del response
            
            return text.strip()
            
        except ValueError as e:
            if "Failed to evaluate chunk: error code 1" in str(e):
                error_msg = (
                    f"[SmartLM] GGUF model ran out of context. "
                    f"Try: 1) Reduce frame_count (currently processing {image.shape[0] if image is not None and len(image.shape) == 4 else 1} frames), "
                    f"2) Increase context_size when loading model (32768 for videos recommended)"
                )
                cstr(error_msg).error.print()
                raise ValueError(error_msg) from e
            else:
                cstr(f"[SmartLM] GGUF generation error: {e}").error.print()
                raise
        except Exception as e:
            cstr(f"[SmartLM] GGUF generation error: {e}").error.print()
            raise
    
    def _generate_llm(self, prompt: str, max_tokens: int, temperature: float,
                      top_p: float, top_k: int, seed: Optional[int], repetition_penalty: float,
                      llm_mode: str, instruction_template: str) -> str:
        # Generate text-only completion with LLM (no images)
        
        # Set seed if provided
        if seed is not None:
            self.model.set_seed(seed)
        
        # Load configuration for the selected mode
        config = LLM_FEW_SHOT_EXAMPLES.get(llm_mode, LLM_FEW_SHOT_EXAMPLES.get("direct_chat", {}))
        system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        examples = config.get("examples", [])
        
        # Get instruction template (custom or from config)
        if instruction_template:
            # Custom instruction provided
            template = instruction_template
        else:
            # Use template from config
            template = config.get("instruction_template", "")
        
        # Build messages based on mode
        if llm_mode != "direct_chat" and template:
            # Apply instruction template with few-shot examples
            req = template.replace("{prompt}", prompt) if "{prompt}" in template else f"{template} {prompt}"
            
            # Build messages: system + examples + user request
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(examples)
            messages.append({"role": "user", "content": req})
        else:
            # Direct chat mode - no instruction wrapper or examples
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        
        # Generate response
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stream=False
            )
            
            # Extract text from response
            text = response['choices'][0]['message']['content']
            return text.strip()
            
        except Exception as e:
            cstr(f"[SmartLM] LLM generation error: {e}").error.print()
            raise
    
    def _generate_florence2(self, image: Any, task_or_prompt: str, max_tokens: int,
                            num_beams: int, do_sample: bool, seed: Optional[int], 
                            repetition_penalty: float = 1.0, text_input: Optional[str] = None,
                            convert_to_bboxes: bool = True) -> tuple[str, dict]:
        # Generate with Florence-2 - returns (text, parsed_data)
        import re
        import torchvision.transforms.functional as F
        
        if seed is not None:
            from transformers import set_seed
            # Hash seed to ensure it's within valid range
            import hashlib
            seed_bytes = str(seed).encode('utf-8')
            hash_object = hashlib.sha256(seed_bytes)
            hashed_seed = int(hash_object.hexdigest(), 16) % (2**32)
            set_seed(hashed_seed)
        
        # Handle image tensor - Florence-2 expects PIL image
        if image is None:
            return ("", {})
        
        # Convert tensor to PIL - handle batch dimension
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                # Batch of images - take first
                image = image[0]
            # Permute from (H, W, C) to (C, H, W) if needed
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            image_pil = F.to_pil_image(image)
        else:
            image_pil = self.tensor_to_pil(image)
        
        if not image_pil:
            return ("", {})
        
        # Get Florence-2 task prompt (just the task token like <CAPTION_TO_PHRASE_GROUNDING>)
        florence_task = FLORENCE_TASKS.get(task_or_prompt, {})
        if florence_task:
            task_prompt = florence_task.get("prompt", task_or_prompt)
        else:
            task_prompt = task_or_prompt
        
        # Build full prompt for generation (task token + text input)
        if text_input and text_input.strip():
            # Tasks like <CAPTION_TO_PHRASE_GROUNDING> expect text after the task token with a space
            prompt = f"{task_prompt} {text_input.strip()}"
        else:
            prompt = task_prompt
        
        # HYBRID APPROACH: Handle device management based on quantization
        device = next(self.model.parameters()).device
        offload_device = device  # Default: same as current device
        
        if hasattr(self, 'is_quantized') and not self.is_quantized:
            # Non-quantized: Use ComfyUI device management
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
                offload_device = mm.unet_offload_device()
                self.model.to(device)
            except:
                pass
        # Quantized: Model stays where device_map placed it (device already set above)
        
        dtype = self.dtype if hasattr(self, 'dtype') else torch.float16
        
        # Process image with do_rescale=False (ComfyUI images are already 0-1)
        inputs = self.processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(dtype).to(device) if torch.is_tensor(v) and v.dtype.is_floating_point else v.to(device)
                  for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=False,
        )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse structured output (bounding boxes, labels, etc.) BEFORE cleaning
        parsed_data = {}
        try:
            W, H = image_pil.size
            # IMPORTANT: Use task_prompt (just the task token) for post-processing, NOT the full prompt
            parsed_answer = self.processor.post_process_generation(
                results, 
                task=task_prompt, 
                image_size=(W, H)
            )
            
            # Extract data for the specific task - Florence-2 returns dict with task token as key
            if isinstance(parsed_answer, dict) and task_prompt in parsed_answer:
                parsed_data = parsed_answer[task_prompt]
                
                # Check if processor returned proper dict with bboxes or just a string
                if isinstance(parsed_data, str):
                    # Custom models may not parse properly, do manual parsing
                    parsed_data = self._parse_florence_location_tokens(parsed_data, W, H)
                
                # Optionally convert quad_boxes and polygons to normalized bboxes
                if isinstance(parsed_data, dict):
                    if convert_to_bboxes:
                        # Convert quad_boxes (8 coords) to bboxes (4 coords) using min/max
                        if 'quad_boxes' in parsed_data and parsed_data['quad_boxes']:
                            bboxes = []
                            for quad in parsed_data['quad_boxes']:
                                # Extract x and y coordinates
                                x_coords = [quad[i] for i in range(0, 8, 2)]
                                y_coords = [quad[i] for i in range(1, 8, 2)]
                                # Get bounding box
                                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                bboxes.append(bbox)
                            # Preserve key order: remove quad_boxes first, then add bboxes at the same position
                            labels = parsed_data.pop('labels', None)
                            parsed_data.pop('quad_boxes')
                            parsed_data['bboxes'] = bboxes
                            if labels is not None:
                                parsed_data['labels'] = labels
                            cstr(f"[SmartLM] Converted {len(bboxes)} quad_boxes to bboxes").msg.print()
                        
                        # Convert polygons to bboxes using min/max
                        elif 'polygons' in parsed_data and parsed_data['polygons']:
                            bboxes = []
                            for polygon in parsed_data['polygons']:
                                x_coords = [pt[0] for pt in polygon]
                                y_coords = [pt[1] for pt in polygon]
                                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                bboxes.append(bbox)
                            # Preserve key order: remove polygons first, then add bboxes at the same position
                            labels = parsed_data.pop('labels', None)
                            parsed_data.pop('polygons')
                            parsed_data['bboxes'] = bboxes
                            if labels is not None:
                                parsed_data['labels'] = labels
                            cstr(f"[SmartLM] Converted {len(bboxes)} polygons to bboxes").msg.print()
                    
                    # Log detection count
                    if 'bboxes' in parsed_data:
                        cstr(f"[SmartLM] Detected {len(parsed_data['bboxes'])} bboxes").msg.print()
                    elif 'quad_boxes' in parsed_data:
                        cstr(f"[SmartLM] Detected {len(parsed_data['quad_boxes'])} quad_boxes").msg.print()
                    elif 'polygons' in parsed_data:
                        cstr(f"[SmartLM] Detected {len(parsed_data['polygons'])} polygons").msg.print()
        except Exception as e:
            cstr(f"[SmartLM] Could not parse Florence-2 output: {e}").error.print()
            parsed_data = {}
        
        # Clean up special tokens for text output
        clean_results = results.replace('</s>', '').replace('<s>', '')
        clean_results = re.sub(r'<[^>]*>', '', clean_results).strip()
        
        # HYBRID APPROACH: Offload non-quantized models back
        if hasattr(self, 'is_quantized') and not self.is_quantized and offload_device != device:
            try:
                self.model.to(offload_device)
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except:
                pass
        
        return (clean_results, parsed_data)
    
    def _parse_florence_location_tokens(self, text: str, width: int, height: int) -> dict:
        # Parse Florence-2 location tokens manually when processor doesn't parse properly.
        #
        # Supports multiple formats:
        # - Bboxes: "label<loc_x1><loc_y1><loc_x2><loc_y2>" (4 tokens)
        # - Polygons: "label<loc_x1><loc_y1><loc_x2><loc_y2>...<loc_xn><loc_yn>" (4+ tokens, multiples of 2)
        # - Quad boxes: "label<loc_x1><loc_y1><loc_x2><loc_y2><loc_x3><loc_y3><loc_x4><loc_y4>" (8 tokens for OCR)
        #
        # Location tokens are normalized to 0-999 range.
        import re
        
        # Pattern to match label followed by location tokens
        # Captures label and all subsequent <loc_###> tokens
        pattern = r'([^<]+?)((?:<loc_\d+>)+)'
        matches = re.findall(pattern, text)
        
        if not matches:
            return {}
        
        bboxes = []
        labels = []
        polygons = []
        quad_boxes = []
        
        for match in matches:
            label = match[0].strip()
            if not label:
                continue
                
            # Extract all location token values
            loc_tokens = re.findall(r'<loc_(\d+)>', match[1])
            if not loc_tokens:
                continue
            
            # Convert normalized coordinates (0-999) to actual pixel coordinates
            coords = [int(token) * width / 1000 if i % 2 == 0 else int(token) * height / 1000 
                     for i, token in enumerate(loc_tokens)]
            
            num_coords = len(coords)
            
            if num_coords == 4:
                # Standard bbox: [x1, y1, x2, y2]
                bboxes.append(coords)
                labels.append(label)
            elif num_coords == 8:
                # Quad box (for OCR): [x1, y1, x2, y2, x3, y3, x4, y4]
                quad_boxes.append(coords)
                if not labels:  # Only append if not already tracking labels
                    labels.append(label)
            elif num_coords > 4 and num_coords % 2 == 0:
                # Polygon: multiple x,y pairs
                # Convert to list of [x,y] pairs
                polygon_points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
                polygons.append(polygon_points)
                if not labels:  # Only append if not already tracking labels
                    labels.append(label)
        
        # Return appropriate format based on what was found
        result = {}
        if bboxes:
            result['bboxes'] = bboxes
            result['labels'] = labels
        if quad_boxes:
            result['quad_boxes'] = quad_boxes
            if 'labels' not in result:
                result['labels'] = labels
        if polygons:
            result['polygons'] = polygons  # type: ignore[assignment]
            if 'labels' not in result:
                result['labels'] = labels
        
        return result
    
    def _draw_bboxes(self, image: Any, data: dict) -> Any:
        # Draw bounding boxes, quad boxes, and polygons on image and return as tensor
        import torch
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Convert tensor to PIL
        if image is None:
            return torch.zeros((1, 64, 64, 3))
        
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # Take first frame if batch
            # Convert from ComfyUI format (H, W, C) in range [0, 1]
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            pil_image = image
        
        # Create a copy to draw on
        draw_image = pil_image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Get detection data
        bboxes = data.get("bboxes", [])
        quad_boxes = data.get("quad_boxes", [])
        polygons: list[list[list[float]]] = data.get("polygons", [])
        labels = data.get("labels", [])
        
        # Try to load font once
        font: Any  # Can be FreeTypeFont or ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw regular bounding boxes
        for i, bbox in enumerate(bboxes):
            # bbox format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label if available
            if i < len(labels):
                label = labels[i]
                # Get text bounding box
                text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1 - 25), label, fill="white", font=font)
        
        # Draw quad boxes (for OCR/text detection)
        for i, quad_box in enumerate(quad_boxes):
            # quad_box format: [x1, y1, x2, y2, x3, y3, x4, y4] - 4 corner points
            # Draw polygon connecting all 4 corners
            points = [(quad_box[j], quad_box[j+1]) for j in range(0, 8, 2)]
            draw.polygon(points, outline="blue", width=3)
            
            # Draw label if available
            if i < len(labels):
                label = labels[i]
                # Use first point for label position
                x1, y1 = points[0]
                text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(text_bbox, fill="blue")
                draw.text((x1, y1 - 25), label, fill="white", font=font)
        
        # Draw polygons
        for i, polygon in enumerate(polygons):
            # polygon format: [[x1, y1], [x2, y2], ..., [xn, yn]]
            points = [tuple(point) for point in polygon]  # type: ignore[misc]
            draw.polygon(points, outline="green", width=3)
            
            # Draw label if available
            if i < len(labels):
                label = labels[i]
                # Use first point for label position
                x1, y1 = points[0]
                text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(text_bbox, fill="green")
                draw.text((x1, y1 - 25), label, fill="white", font=font)
        
        # Convert back to tensor format
        img_array = np.array(draw_image).astype(np.float32) / 255.0
        tensor_out = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
        
        return tensor_out
    

