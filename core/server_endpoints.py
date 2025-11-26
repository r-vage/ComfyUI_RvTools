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

# Eclipse Wildcard Processor - Server Endpoints
#
# Provides REST API endpoints for wildcard management:
# - GET /eclipse/wildcards/list - Get list of available wildcards
# - GET /eclipse/wildcards/refresh - Reload wildcards from disk
# - POST /eclipse/wildcards - Process text with wildcards

import json
import os
import sys
from typing import Dict, Any, List, Optional

from server import PromptServer
from aiohttp import web

from .common import cstr
from .wildcard_engine import (get_wildcard_list, wildcard_load, process)


class WildcardEndpoints:
    # Manages wildcard server endpoints.

    def __init__(self, wildcard_path: Optional[str] = None):
        #nitialize endpoints.
        # 
        # Args:
        #     wildcard_path: Path to wildcard directory. If None, uses default.
        if wildcard_path is None:
            wildcard_path = self._get_default_wildcard_path()
        
        self.wildcard_path = wildcard_path
        
        # Load wildcards on initialization
        cstr(f"[Wildcard] Loading wildcards from: {wildcard_path}").msg.print()
        wildcard_load(wildcard_path)
        
        self._register_endpoints()
    
    def _get_default_wildcard_path(self) -> str:
        # Determine the default wildcard path.
        # 
        # Priority:
        # 1. ComfyUI/models/wildcards (create if doesn't exist and copy examples)
        # 2. Extension's wildcards/ folder (fallback)
        # 
        # Returns:
        #     Path to wildcard directory
        # Extension's wildcard folder (fallback)
        extension_root = os.path.dirname(os.path.dirname(__file__))
        extension_wildcard_path = os.path.join(extension_root, "wildcards")
        
        # Try to find ComfyUI root (go up from custom_nodes/ComfyUI_Eclipse_X)
        comfyui_root = os.path.abspath(os.path.join(extension_root, "..", ".."))
        models_wildcard_path = os.path.join(comfyui_root, "models", "wildcards")
        
        # Check if we're actually in a ComfyUI installation
        if os.path.exists(os.path.join(comfyui_root, "models")):
            # Create models/wildcards directory if it doesn't exist
            if not os.path.exists(models_wildcard_path):
                try:
                    os.makedirs(models_wildcard_path, exist_ok=True)
                    cstr(f"[Wildcard] Created directory: {models_wildcard_path}").msg.print()
                    
                    # Copy example files from extension's wildcards folder
                    if os.path.exists(extension_wildcard_path):
                        self._copy_example_wildcards(extension_wildcard_path, models_wildcard_path)
                except Exception as e:
                    cstr(f"[Wildcard] Failed to create {models_wildcard_path}: {e}").error.print()
                    return extension_wildcard_path
            
            return models_wildcard_path
        else:
            # Not in a standard ComfyUI structure, use extension folder
            cstr("[Wildcard] Using extension's wildcard folder (ComfyUI models dir not found)").msg.print()
            return extension_wildcard_path
    
    def _copy_example_wildcards(self, source_dir: str, dest_dir: str) -> None:
        # Copy example wildcard files from source to destination.
        # 
        # Args:
        #     source_dir: Source directory with example wildcards
        #     dest_dir: Destination directory
        import shutil
        
        try:
            copied_count = 0
            for filename in os.listdir(source_dir):
                if filename.endswith(('.txt', '.yaml', '.yml')):
                    source_file = os.path.join(source_dir, filename)
                    dest_file = os.path.join(dest_dir, filename)
                    
                    # Only copy if destination doesn't exist
                    if not os.path.exists(dest_file):
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
            
            if copied_count > 0:
                cstr(f"[Wildcard] Copied {copied_count} example wildcard files to {dest_dir}").msg.print()
        except Exception as e:
            cstr(f"[Wildcard] Error copying example wildcards: {e}").error.print()

    def _register_endpoints(self):
        # Register all endpoints with PromptServer.
        
        @PromptServer.instance.routes.get("/eclipse/wildcards/list")
        async def handle_get_wildcard_list(request):
            # GET /eclipse/wildcards/list
            # 
            # Returns:
            #     JSON list of available wildcards in format: ['__keyword1__', '__keyword2__', ...]
            try:
                wildcard_list = get_wildcard_list()
                return web.json_response(wildcard_list)
            except Exception as e:
                cstr(f"[Wildcard] Error getting wildcard list: {e}").error.print()
                return web.json_response([])

        @PromptServer.instance.routes.get("/eclipse/wildcards/refresh")
        async def handle_refresh_wildcards(request):
            # GET /eclipse/wildcards/refresh
            # 
            # Reloads wildcards from disk. Useful for discovering newly added wildcard files.
            # 
            # Returns:
            #     JSON with success status and count of loaded wildcards
            try:
                wildcard_load(self.wildcard_path)
                wildcard_list = get_wildcard_list()
                
                return web.json_response({
                    "success": True,
                    "message": f"Loaded {len(wildcard_list)} wildcard groups",
                    "count": len(wildcard_list)
                })
            except Exception as e:
                cstr(f"[Wildcard] Error refreshing wildcards: {e}").error.print()
                return web.json_response({
                    "success": False,
                    "message": str(e),
                    "count": 0
                })

        @PromptServer.instance.routes.post("/eclipse/wildcards/process")
        async def handle_process_wildcards(request):
            # POST /eclipse/wildcards/process
            # 
            # Process text with wildcard expansion.
            # 
            # Request JSON:
            # {
            #     "text": "Text with __wildcards__ and {options|go|here}",
            #     "seed": 12345 (optional)
            # }
            # 
            # Returns:
            #     JSON with processed text
            try:
                # Parse request body
                if request.content_length:
                    body = await request.json()
                else:
                    body = {}

                text = body.get("text", "")
                seed = body.get("seed", None)

                if not isinstance(text, str):
                    return web.json_response({
                        "success": False,
                        "error": "Invalid text parameter"
                    })

                # Process the text
                result = process(text, seed=seed)

                return web.json_response({
                    "success": True,
                    "input": text,
                    "output": result,
                    "seed": seed
                })

            except Exception as e:
                cstr(f"[Wildcard] Error processing wildcards: {e}").error.print()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @PromptServer.instance.routes.post("/eclipse/smartlm/search_model")
        async def handle_search_model(request):
            # POST /eclipse/smartlm/search_model
            # 
            # Search for locally downloaded model based on repo_id.
            # 
            # Request JSON:
            # {
            #     "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            #     "model_type": "qwenvl"
            # }
            # 
            # Returns:
            #     JSON with found status and local_path if found
            try:
                body = await request.json()
                repo_id = body.get("repo_id", "")
                model_type = body.get("model_type", "")
                
                if not repo_id:
                    return web.json_response({
                        "found": False,
                        "error": "repo_id is required"
                    })
                
                # Extract model/file name from repo_id
                # Handle different formats:
                # 1. "Qwen/Qwen3-VL-4B-Instruct" -> "Qwen3-VL-4B-Instruct" (folder)
                # 2. "https://huggingface.co/author/repo/resolve/main/file.gguf" -> "file.gguf" (file)
                model_name = repo_id.split('/')[-1]
                
                # Try to find the model in the LLM folder
                import folder_paths
                from pathlib import Path
                from .smartlm_files import search_model_file
                
                llm_base = Path(folder_paths.models_dir) / "LLM"
                
                if not llm_base.exists():
                    return web.json_response({
                        "found": False,
                        "error": "LLM folder not found"
                    })
                
                found_path = None
                
                # Strategy 1: Direct folder match by model_name
                candidate = llm_base / model_name
                if candidate.exists() and candidate.is_dir():
                    # Verify it has model files
                    model_files = list(candidate.glob('*.safetensors')) + list(candidate.glob('*.bin')) + list(candidate.glob('*.gguf'))
                    if model_files:
                        found_path = f"{model_name}/"
                
                # Strategy 2: Search for exact filename using search_model_file
                if not found_path:
                    found_file = search_model_file(model_name, llm_base)
                    if found_file:
                        # Return relative path from llm_base
                        relative_path = found_file.relative_to(llm_base)
                        found_path = relative_path.as_posix()
                
                # Strategy 3: Partial name match (for repo names without exact file)
                if not found_path and repo_id.startswith('http'):
                    # Extract repo name from URL (e.g., "Qwen2.5-VL-7B-Abliterated-Caption-it-GGUF")
                    parts = repo_id.split('/')
                    if len(parts) >= 5:
                        repo_name = parts[4]
                        search_name = repo_name.replace('-GGUF', '').replace('-gguf', '')
                        
                        # Search for folders matching the repo name
                        for folder in llm_base.rglob('*'):
                            if folder.is_dir() and search_name.lower() in folder.name.lower():
                                model_files = list(folder.glob('*.safetensors')) + list(folder.glob('*.bin')) + list(folder.glob('*.gguf'))
                                if model_files:
                                    relative_path = folder.relative_to(llm_base)
                                    found_path = relative_path.as_posix() + "/"
                                    break
                
                if found_path:
                    return web.json_response({
                        "found": True,
                        "local_path": found_path,
                        "repo_id": repo_id
                    })
                else:
                    return web.json_response({
                        "found": False
                    })
                
            except Exception as e:
                cstr(f"[SmartLM] Error searching for model: {e}").error.print()
                return web.json_response({
                    "found": False,
                    "error": str(e)
                })

        @PromptServer.instance.routes.post("/eclipse/smartlm/search_mmproj")
        async def handle_search_mmproj(request):
            # POST /eclipse/smartlm/search_mmproj
            # 
            # Search for mmproj file in the same folder as the model.
            # This handles cases where mmproj filename differs from template URL.
            # 
            # Request JSON:
            # {
            #     "model_path": "Qwen-VL/Model/model.gguf",
            #     "mmproj_url": "https://huggingface.co/.../mmproj.gguf" (optional)
            # }
            # 
            # Returns:
            #     JSON with found status and local_path if found
            try:
                body = await request.json()
                model_path = body.get("model_path", "")
                mmproj_url = body.get("mmproj_url", "")
                
                if not model_path:
                    return web.json_response({
                        "found": False,
                        "error": "model_path is required"
                    })
                
                import folder_paths
                from pathlib import Path
                llm_base = Path(folder_paths.models_dir) / "LLM"
                
                if not llm_base.exists():
                    return web.json_response({
                        "found": False,
                        "error": "LLM folder not found"
                    })
                
                # Get the folder containing the model
                model_full_path = llm_base / model_path
                
                if model_full_path.is_file():
                    # Model is a file, search in its parent folder
                    model_folder = model_full_path.parent
                else:
                    # Model is a folder
                    model_folder = model_full_path
                
                if not model_folder.exists():
                    return web.json_response({
                        "found": False,
                        "error": "Model folder not found"
                    })
                
                # Search for mmproj files in the model folder
                mmproj_files = []
                for file in model_folder.iterdir():
                    if file.is_file():
                        # Match .mmproj files or .gguf files with 'mmproj' in name
                        if file.suffix == '.mmproj' or (file.suffix == '.gguf' and 'mmproj' in file.name.lower()):
                            mmproj_files.append(file)
                
                if not mmproj_files:
                    return web.json_response({
                        "found": False
                    })
                
                # If multiple mmproj files, try to match by name from URL
                selected_mmproj = mmproj_files[0]  # Default to first
                
                if len(mmproj_files) > 1 and mmproj_url:
                    # Extract filename from URL
                    url_filename = mmproj_url.split('/')[-1]
                    # Try to find best match
                    for mmproj_file in mmproj_files:
                        if url_filename.lower() in mmproj_file.name.lower():
                            selected_mmproj = mmproj_file
                            break
                
                # Return relative path from llm_base
                relative_path = selected_mmproj.relative_to(llm_base)
                found_path = relative_path.as_posix()
                
                return web.json_response({
                    "found": True,
                    "local_path": found_path
                })
                
            except Exception as e:
                cstr(f"[SmartLM] Error searching for mmproj: {e}").error.print()
                return web.json_response({
                    "found": False,
                    "error": str(e)
                })

        @PromptServer.instance.routes.get("/eclipse/smartlml_advanced_defaults")
        async def handle_get_smartlm_defaults(request):
            # GET /eclipse/smartlml_advanced_defaults
            # 
            # Returns Smart LML advanced parameter defaults from config file.
            # 
            # Returns:
            #     JSON with default parameters for each model type
            try:
                import folder_paths
                # Check Eclipse folder first, fallback to repo
                eclipse_config = os.path.join(folder_paths.models_dir, "Eclipse", "config", "smartlm_advanced_defaults.json")
                extension_root = os.path.dirname(os.path.dirname(__file__))
                repo_config = os.path.join(extension_root, "templates", "config", "smartlm_advanced_defaults.json")
                config_path = eclipse_config if os.path.exists(eclipse_config) else repo_config
                
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        defaults = json.load(f)
                    return web.json_response(defaults)
                else:
                    # Return built-in defaults if config doesn't exist
                    return web.json_response({
                        "QwenVL": {
                            "device": "cuda",
                            "use_torch_compile": False,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 50,
                            "num_beams": 3,
                            "do_sample": True,
                            "repetition_penalty": 1.0,
                            "frame_count": 8
                        },
                        "Florence2": {
                            "device": "cuda",
                            "use_torch_compile": False,
                            "num_beams": 3,
                            "do_sample": True
                        },
                        "LLM": {
                            "temperature": 1.0,
                            "top_p": 0.9,
                            "top_k": 50,
                            "repetition_penalty": 1.2
                        }
                    })
            except Exception as e:
                cstr(f"[SmartLM] Error loading advanced defaults: {e}").error.print()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        cstr("[Wildcard] Registered server endpoints").msg.print()


def onprompt_populate_wildcards(json_data):
    # Preprocess wildcard nodes before execution.
    # 
    # This runs BEFORE ComfyUI's execution engine, allowing us to:
    # 1. Detect seed connections in the prompt
    # 2. Extract actual seed values from connected nodes
    # 3. Process wildcards with the correct seed
    # 4. Update the prompt with processed text
    # 5. Does NOT switch mode or send UI feedback (for realtime preview support)
    prompt = json_data.get('prompt', {})
    
    for node_id, node_data in prompt.items():
        # Check if this is a Wildcard Processor node (old version)
        if 'class_type' not in node_data:
            continue
            
        if node_data['class_type'] != 'Wildcard Processor [Eclipse]':
            continue
        
        inputs = node_data.get('inputs', {})
        mode = inputs.get('mode', 'populate')
        
        # In fixed mode, normalize the seed to 0 to ensure caching works
        # The seed is not used for wildcard processing in fixed mode
        if mode == 'fixed':
            # Force seed to 0 in fixed mode so cache works regardless of seed changes
            inputs['seed'] = 0
            continue
        
        # Only process wildcards in populate mode
        if mode != 'populate':
            continue
        
        wildcard_text = inputs.get('wildcard_text', '')
        if not wildcard_text or not isinstance(wildcard_text, str):
            continue
        
        # Get seed - check if it's connected (list format) or widget value (int)
        seed_value = inputs.get('seed', 0)
        
        if isinstance(seed_value, list):
            # Seed is connected - extract actual value from connected node
            try:
                connected_node_id = str(seed_value[0])
                connected_node = prompt.get(connected_node_id)
                
                if not connected_node:
                    cstr(f"[Wildcard] Connected seed node {connected_node_id} not found").warning.print()
                    continue
                
                class_type = connected_node.get('class_type', '')
                connected_inputs = connected_node.get('inputs', {})
                
                # Handle different seed node types (like Impact Pack does)
                if class_type == 'Seed (rgthree)':
                    input_seed = int(connected_inputs.get('seed', 0))
                elif class_type in ['ImpactInt', 'Primitive', 'PrimitiveNode']:
                    input_seed = int(connected_inputs.get('value', 0))
                else:
                    # Try common parameter names
                    input_seed = None
                    for key in ['seed', 'value', 'number', 'int']:
                        if key in connected_inputs:
                            value = connected_inputs[key]
                            if not isinstance(value, list):  # Not another connection
                                input_seed = int(value)
                                break
                    
                    if input_seed is None:
                        cstr(f"[Wildcard] Could not extract seed from node type: {class_type}").warning.print()
                        continue
                
            except Exception as e:
                cstr(f"[Wildcard] Error extracting seed from connection: {e}").error.print()
                continue
        else:
            # Seed is a direct value
            input_seed = int(seed_value)
        
        # Process wildcards with the determined seed
        try:
            processed_text = process(wildcard_text, seed=input_seed)
            
            # Update the populated_text in the prompt (this is what gets sent to execute)
            inputs['populated_text'] = processed_text
            
            # Also update the seed input so execute() receives the actual seed used
            # This ensures the seed is saved correctly in metadata
            inputs['seed'] = input_seed
            
        except Exception as e:
            cstr(f"[Wildcard] Error processing wildcards for node {node_id}: {e}").error.print()
    
    # CRITICAL: Must return json_data for the handler chain to continue
    return json_data


# Initialize endpoints when module is imported
def initialize_endpoints(wildcard_path: Optional[str] = None):
    # Initialize wildcard server endpoints.
    # 
    # Args:
    #     wildcard_path: Path to wildcard directory. If None, uses default.
    try:
        WildcardEndpoints(wildcard_path)
        
        # Register prompt handler for wildcard preprocessing
        PromptServer.instance.add_on_prompt_handler(onprompt_populate_wildcards)
        
        cstr("[Wildcard] Server endpoints and prompt handler initialized successfully").msg.print()
    except Exception as e:
        cstr(f"[Wildcard] Failed to initialize endpoints: {e}").error.print()