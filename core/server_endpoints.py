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
Eclipse Wildcard Processor - Server Endpoints

Provides REST API endpoints for wildcard management:
- GET /eclipse/wildcards/list - Get list of available wildcards
- GET /eclipse/wildcards/refresh - Reload wildcards from disk
- POST /eclipse/wildcards - Process text with wildcards
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

from server import PromptServer
from aiohttp import web

from .wildcard_engine import (get_wildcard_list, wildcard_load, process)


class WildcardEndpoints:
    """Manages wildcard server endpoints."""

    def __init__(self, wildcard_path: Optional[str] = None):
        """
        Initialize endpoints.
        
        Args:
            wildcard_path: Path to wildcard directory. If None, uses default.
        """
        if wildcard_path is None:
            wildcard_path = self._get_default_wildcard_path()
        
        self.wildcard_path = wildcard_path
        
        # Load wildcards on initialization
        logging.info(f"[Eclipse Wildcard] Loading wildcards from: {wildcard_path}")
        wildcard_load(wildcard_path)
        
        self._register_endpoints()
    
    def _get_default_wildcard_path(self) -> str:
        """
        Determine the default wildcard path.
        
        Priority:
        1. ComfyUI/models/wildcards (create if doesn't exist and copy examples)
        2. Extension's wildcards/ folder (fallback)
        
        Returns:
            Path to wildcard directory
        """
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
                    logging.info(f"[Eclipse Wildcard] Created directory: {models_wildcard_path}")
                    
                    # Copy example files from extension's wildcards folder
                    if os.path.exists(extension_wildcard_path):
                        self._copy_example_wildcards(extension_wildcard_path, models_wildcard_path)
                except Exception as e:
                    logging.error(f"[Eclipse Wildcard] Failed to create {models_wildcard_path}: {e}")
                    return extension_wildcard_path
            
            return models_wildcard_path
        else:
            # Not in a standard ComfyUI structure, use extension folder
            logging.info("[Eclipse Wildcard] Using extension's wildcard folder (ComfyUI models dir not found)")
            return extension_wildcard_path
    
    def _copy_example_wildcards(self, source_dir: str, dest_dir: str) -> None:
        """
        Copy example wildcard files from source to destination.
        
        Args:
            source_dir: Source directory with example wildcards
            dest_dir: Destination directory
        """
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
                logging.info(f"[Eclipse Wildcard] Copied {copied_count} example wildcard files to {dest_dir}")
        except Exception as e:
            logging.error(f"[Eclipse Wildcard] Error copying example wildcards: {e}")

    def _register_endpoints(self):
        """Register all endpoints with PromptServer."""
        
        @PromptServer.instance.routes.get("/eclipse/wildcards/list")
        async def handle_get_wildcard_list(request):
            """
            GET /eclipse/wildcards/list
            
            Returns:
                JSON list of available wildcards in format: ['__keyword1__', '__keyword2__', ...]
            """
            try:
                wildcard_list = get_wildcard_list()
                return web.json_response(wildcard_list)
            except Exception as e:
                logging.error(f"[Eclipse Wildcard] Error getting wildcard list: {e}")
                return web.json_response([])

        @PromptServer.instance.routes.get("/eclipse/wildcards/refresh")
        async def handle_refresh_wildcards(request):
            """
            GET /eclipse/wildcards/refresh
            
            Reloads wildcards from disk. Useful for discovering newly added wildcard files.
            
            Returns:
                JSON with success status and count of loaded wildcards
            """
            try:
                wildcard_load(self.wildcard_path)
                wildcard_list = get_wildcard_list()
                
                return web.json_response({
                    "success": True,
                    "message": f"Loaded {len(wildcard_list)} wildcard groups",
                    "count": len(wildcard_list)
                })
            except Exception as e:
                logging.error(f"[Eclipse Wildcard] Error refreshing wildcards: {e}")
                return web.json_response({
                    "success": False,
                    "message": str(e),
                    "count": 0
                })

        @PromptServer.instance.routes.post("/eclipse/wildcards/process")
        async def handle_process_wildcards(request):
            """
            POST /eclipse/wildcards/process
            
            Process text with wildcard expansion.
            
            Request JSON:
            {
                "text": "Text with __wildcards__ and {options|go|here}",
                "seed": 12345 (optional)
            }
            
            Returns:
                JSON with processed text
            """
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
                logging.error(f"[Eclipse Wildcard] Error processing wildcards: {e}")
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        logging.info("[Eclipse Wildcard] Registered server endpoints")


def onprompt_populate_wildcards(json_data):
    """
    Preprocess wildcard nodes before execution (Impact Pack approach).
    
    This runs BEFORE ComfyUI's execution engine, allowing us to:
    1. Detect seed connections in the prompt
    2. Extract actual seed values from connected nodes
    3. Process wildcards with the correct seed
    4. Update the prompt with processed text
    5. Avoid double execution issues
    """
    prompt = json_data.get('prompt', {})
    
    for node_id, node_data in prompt.items():
        # Check if this is a Wildcard Processor node
        if 'class_type' not in node_data:
            continue
            
        if node_data['class_type'] != 'Wildcard Processor [Eclipse]':
            continue
        
        inputs = node_data.get('inputs', {})
        mode = inputs.get('mode', 'populate')
        wildcard_text = inputs.get('wildcard_text', '')
        populated_text = inputs.get('populated_text', '')
        seed = inputs.get('seed', 0)
        
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
                    logging.warning(f"[Eclipse Wildcard] Connected seed node {connected_node_id} not found")
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
                        logging.warning(f"[Eclipse Wildcard] Could not extract seed from node type: {class_type}")
                        continue
                
            except Exception as e:
                logging.error(f"[Eclipse Wildcard] Error extracting seed from connection: {e}")
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
            logging.error(f"[Eclipse Wildcard] Error processing wildcards for node {node_id}: {e}")
    
    # CRITICAL: Must return json_data for the handler chain to continue
    return json_data


# Initialize endpoints when module is imported
def initialize_endpoints(wildcard_path: Optional[str] = None):
    """
    Initialize wildcard server endpoints.
    
    Args:
        wildcard_path: Path to wildcard directory. If None, uses default.
    """
    try:
        WildcardEndpoints(wildcard_path)
        
        # Register prompt handler for wildcard preprocessing (Impact Pack approach)
        PromptServer.instance.add_on_prompt_handler(onprompt_populate_wildcards)
        
        logging.info("[Eclipse Wildcard] Server endpoints and prompt handler initialized successfully")
    except Exception as e:
        logging.error(f"[Eclipse Wildcard] Failed to initialize endpoints: {e}")
