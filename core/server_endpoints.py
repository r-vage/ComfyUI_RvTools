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
    5. Send feedback to frontend to update UI
    """
    from server import PromptServer
    
    prompt = json_data.get('prompt', {})
    updated_widget_values = {}
    
    for node_id, node_data in prompt.items():
        # Check if this is a Wildcard Processor node
        if 'class_type' not in node_data:
            continue
            
        if node_data['class_type'] != 'Wildcard Processor [Eclipse]':
            continue
        
        inputs = node_data.get('inputs', {})
        
        # Legacy adapter - convert boolean mode to string
        if isinstance(inputs.get('mode'), bool):
            inputs['mode'] = 'populate' if inputs['mode'] else 'fixed'
        
        # Legacy adapter - convert old 'reproduce' mode to 'fixed' and update UI
        if inputs.get('mode') == 'reproduce':
            inputs['mode'] = 'fixed'
            # Send feedback to update the UI widget
            PromptServer.instance.send_sync("eclipse-node-feedback", {
                "node_id": node_id,
                "widget_name": "mode",
                "type": "STRING",
                "value": 'fixed'
            })
        
        mode = inputs.get('mode', 'populate')
        
        # Only process in populate mode
        if mode == 'populate' and isinstance(inputs.get('populated_text'), str):
            wildcard_text = inputs.get('wildcard_text', '')
            if not wildcard_text:
                continue
            
            # Get seed - handle connections and direct values
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
                    
                    # Handle different seed node types (same as Impact Pack)
                    if class_type == 'Seed (rgthree)':
                        input_seed = int(connected_inputs.get('seed', 0))
                    elif class_type == 'ImpactInt':
                        input_seed = int(connected_inputs.get('value', 0))
                    elif class_type in ['Primitive', 'PrimitiveNode']:
                        input_seed = int(connected_inputs.get('value', 0))
                    else:
                        # Try to find seed value from common parameter names
                        input_seed = None
                        for key in ['seed', 'value', 'int', 'number']:
                            if key in connected_inputs:
                                value = connected_inputs[key]
                                if not isinstance(value, list):  # Not another connection
                                    input_seed = int(value)
                                    break
                        
                        if input_seed is None:
                            logging.info(f"[Eclipse Wildcard] Only `ImpactInt`, `Seed (rgthree)` and `Primitive` Node are allowed as the seed. It will be ignored.")
                            continue
                    
                except Exception:
                    continue
            else:
                # Seed is a direct value
                input_seed = int(seed_value)
            
            # Process wildcards with the determined seed
            try:
                processed_text = process(wildcard_text, seed=input_seed)
                
                # Update the populated_text in the prompt
                inputs['populated_text'] = processed_text
                
                # Switch mode to 'fixed' to preserve the result on reload
                inputs['mode'] = 'fixed'
                
                # Send feedback to frontend to update both widgets
                PromptServer.instance.send_sync("eclipse-node-feedback", {
                    "node_id": node_id,
                    "widget_name": "populated_text",
                    "type": "STRING",
                    "value": processed_text
                })
                
                PromptServer.instance.send_sync("eclipse-node-feedback", {
                    "node_id": node_id,
                    "widget_name": "mode",
                    "type": "STRING",
                    "value": 'fixed'
                })
                
                # Track updated values for workflow metadata
                updated_widget_values[node_id] = processed_text
                
            except Exception as e:
                logging.error(f"[Eclipse Wildcard] Error processing wildcards for node {node_id}: {e}")
    
    # Update workflow metadata (extra_pnginfo) with the processed values
    if 'extra_data' in json_data and 'extra_pnginfo' in json_data['extra_data']:
        if 'workflow' in json_data['extra_data']['extra_pnginfo']:
            for node in json_data['extra_data']['extra_pnginfo']['workflow']['nodes']:
                key = str(node['id'])
                if key in updated_widget_values:
                    # Update populated_text widget value (index 1)
                    node['widgets_values'][1] = updated_widget_values[key]
                    # Set mode to 'fixed' (index 2)
                    node['widgets_values'][2] = 'fixed'
    
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
