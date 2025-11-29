# Comfyui_Eclipse Extension Loader
#
# Initializes and loads all custom nodes for Comfyui_Eclipse, providing NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS for the extension.
#
# Author: r-vage
#
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
WEB_DIRECTORY = "./js"

import importlib.util
import os
import re
import json
import __main__
from .core import version, cstr
from aiohttp import web 
import server
import folder_paths

from typing import Any, Dict, Type

NODE_CLASS_MAPPINGS: Dict[str, Type[Any]] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# MESSAGE TEMPLATES
cstr.color.add_code("msg", f"{cstr.color.LIGHTGREEN}Eclipse: {cstr.color.END}")
cstr.color.add_code("warning", f"{cstr.color.LIGHTGREEN}Eclipse {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.color.add_code("debug", f"{cstr.color.LIGHTGREEN}Eclipse {cstr.color.LIGHTBEIGE}Debug: {cstr.color.END}")
cstr.color.add_code("error", f"{cstr.color.RED}Eclipse {cstr.color.END}Error: {cstr.color.END}")

cstr(f'Version: {version}').msg.print()

# Early check of wrappers (for consistent startup logging)
try:
    from .core import gguf_wrapper
except Exception as e:
    cstr(f"[GGUF Wrapper] Failed to load: {e}").warning.print()

try:
    from .core import nunchaku_wrapper
except Exception as e:
    cstr(f"[Nunchaku Wrapper] Failed to load: {e}").warning.print()

try:
    from .core import florence2_wrapper
    # Show tip if using fallback
    if not florence2_wrapper.FLORENCE2_CUSTOM_AVAILABLE:
        cstr(f"[Florence-2 Wrapper] Tip: Install comfyui-florence2 extension for better compatibility").msg.print()
except Exception as e:
    cstr(f"[Florence-2 Wrapper] Failed to load: {e}").warning.print()

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)
    dir = os.path.abspath(dir)
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Initialize Eclipse folder structure with templates (one-time copy on first run)
from .core.common import copy_prompt_files_once, create_junction, migrate_old_folders
import sys

comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
eclipse_dir = os.path.join(comfyui_root, 'models', 'Eclipse')

# Migrate user files from old locations (one-time migration)
migrate_old_folders(comfyui_root)

# Copy templates to models/Eclipse/ structure
repo_templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
repo_prompt_dir = os.path.join(repo_templates_dir, 'prompt')
repo_loader_dir = os.path.join(repo_templates_dir, 'loader_templates')
repo_smartlm_dir = os.path.join(repo_templates_dir, 'smartlm_templates')
repo_config_dir = os.path.join(repo_templates_dir, 'config')

eclipse_prompt_dir = os.path.join(eclipse_dir, 'smart_prompt')
eclipse_loader_dir = os.path.join(eclipse_dir, 'loader_templates')
eclipse_smartlm_dir = os.path.join(eclipse_dir, 'smartlm_templates')
eclipse_config_dir = os.path.join(eclipse_dir, 'config')

# Check if force_update is enabled in config
import json
force_update = False
config_file = os.path.join(repo_config_dir, 'smartlm_prompt_defaults.json')
if os.path.exists(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            force_update = config_data.get('_force_update', False)
    except:
        pass

# One-time copy of templates to Eclipse folder (smart_prompt and loader - normal behavior)
if not os.path.exists(eclipse_prompt_dir) and os.path.exists(repo_prompt_dir):
    copy_prompt_files_once(repo_prompt_dir, eclipse_prompt_dir)

if not os.path.exists(eclipse_loader_dir) and os.path.exists(repo_loader_dir):
    copy_prompt_files_once(repo_loader_dir, eclipse_loader_dir)

# smartlm_templates: copy on first run (folder doesn't exist) OR force update (overwrite existing)
if not os.path.exists(eclipse_smartlm_dir):
    # First run: copy templates from repo
    if os.path.exists(repo_smartlm_dir):
        copy_prompt_files_once(repo_smartlm_dir, eclipse_smartlm_dir)
elif force_update:
    # Force update: only update templates that exist in repo (preserve user templates)
    import shutil
    try:
        # Get list of template files in repo
        repo_templates = set()
        if os.path.exists(repo_smartlm_dir):
            repo_templates = {item for item in os.listdir(repo_smartlm_dir) 
                            if os.path.isfile(os.path.join(repo_smartlm_dir, item)) and item.endswith('.json')}
        
        # Only delete templates that exist in repo (user templates are preserved)
        deleted_count = 0
        for item in os.listdir(eclipse_smartlm_dir):
            if item in repo_templates:
                item_path = os.path.join(eclipse_smartlm_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    deleted_count += 1
        
        # Copy all templates from repo
        copied_count = 0
        for item in repo_templates:
            src = os.path.join(repo_smartlm_dir, item)
            dst = os.path.join(eclipse_smartlm_dir, item)
            shutil.copy2(src, dst)
            copied_count += 1
        
        cstr(f"[Eclipse] Force updated {copied_count} repo template(s), preserved user templates").msg.print()
    except Exception as e:
        cstr(f"[Eclipse] Warning: Could not fully update templates: {e}").warning.print()

# Note: smartlm_prompt_defaults.json is always loaded from repo folder
# Other config files: copy on first run OR force update if flag is set
if force_update or not os.path.exists(eclipse_config_dir):
    os.makedirs(eclipse_config_dir, exist_ok=True)
    if force_update and os.path.exists(repo_config_dir):
        # Force update: overwrite config files (except smartlm_prompt_defaults.json)
        import shutil
        for item in os.listdir(repo_config_dir):
            if item != 'smartlm_prompt_defaults.json':  # Skip this one, always loaded from repo
                src = os.path.join(repo_config_dir, item)
                dst = os.path.join(eclipse_config_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        cstr("[Eclipse] Force updated config files").msg.print()

# Reset force_update flag after updates are complete
if force_update:
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config_data['_force_update'] = False
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        cstr(f"[Eclipse] Warning: Could not reset _force_update flag: {e}").warning.print()

# Create junction for wildcards/smart_prompt → Eclipse/smart_prompt (no duplication)
wildcards_smartprompt_dir = os.path.join(comfyui_root, 'models', 'wildcards', 'smart_prompt')
if not os.path.exists(wildcards_smartprompt_dir) and os.path.exists(eclipse_prompt_dir):
    create_junction(eclipse_prompt_dir, wildcards_smartprompt_dir)

# Update references to use Eclipse folder
models_smartprompt_dir = eclipse_prompt_dir  # For API compatibility
models_loader_dir = eclipse_loader_dir
repo_prompt_dir = eclipse_prompt_dir  # Fallback uses Eclipse copy

# API route for serving loader templates
@server.PromptServer.instance.routes.get("/eclipse/loader_templates/{filename}")
async def serve_loader_template(request):
    filename = request.match_info.get('filename', '')
    if not filename.endswith('.json'):
        return web.Response(status=400, text="Invalid file type")
    
    # Serve from primary models folder
    template_dir = models_loader_dir
    template_path = os.path.join(template_dir, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(template_path).startswith(os.path.abspath(template_dir)):
        return web.Response(status=403, text="Access denied")
    
    if os.path.exists(template_path) and os.path.isfile(template_path):
        return web.FileResponse(template_path)
    else:
        return web.Response(status=404, text="Template not found")

# API route for getting template list
@server.PromptServer.instance.routes.get("/eclipse/loader_templates_list")
async def get_loader_templates_list(request):
    from .py.RvLoader_SmartLoader import get_template_list
    templates = get_template_list()
    return web.json_response(templates)

# Smart LML Templates API (unified QwenVL + Florence-2)
@server.PromptServer.instance.routes.get("/eclipse/smartlm_templates/{filename}")
async def serve_smartlm_template(request):
    filename = request.match_info.get('filename', '')
    if not filename.endswith('.json'):
        return web.Response(status=400, text="Invalid file type")
    
    template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'smartlm_templates')
    template_path = os.path.join(template_dir, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(template_path).startswith(os.path.abspath(template_dir)):
        return web.Response(status=403, text="Access denied")
    
    if os.path.exists(template_path) and os.path.isfile(template_path):
        return web.FileResponse(template_path)
    else:
        return web.Response(status=404, text="Template not found")

@server.PromptServer.instance.routes.get("/eclipse/smartlm_templates_list")
async def get_smartlm_templates_list(request):
    from .core.smartlm_base import get_template_list
    templates = get_template_list()
    return web.json_response(templates)

@server.PromptServer.instance.routes.post("/eclipse/smartlm_templates/{filename}")
async def update_smartlm_template(request):
    """Update template settings from frontend (only visible widget values)"""
    filename = request.match_info.get('filename', '')
    if not filename.endswith('.json'):
        return web.Response(status=400, text="Invalid file type")
    
    # Priority: Eclipse models folder first, then fall back to repo folder
    eclipse_template_dir = os.path.join(folder_paths.models_dir, "Eclipse", "smartlm_templates")
    repo_template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'smartlm_templates')
    
    # Find which location the template exists in
    eclipse_template_path = os.path.join(eclipse_template_dir, filename)
    repo_template_path = os.path.join(repo_template_dir, filename)
    
    # Security: prevent directory traversal
    if not (os.path.abspath(eclipse_template_path).startswith(os.path.abspath(eclipse_template_dir)) or
            os.path.abspath(repo_template_path).startswith(os.path.abspath(repo_template_dir))):
        return web.Response(status=403, text="Access denied")
    
    # Determine which path to use (prefer Eclipse folder for writing)
    template_path = None
    if os.path.exists(eclipse_template_path):
        # Template exists in Eclipse folder - update it there
        template_path = eclipse_template_path
    elif os.path.exists(repo_template_path):
        # Template exists only in repo - copy to Eclipse folder first, then update
        os.makedirs(eclipse_template_dir, exist_ok=True)
        import shutil
        shutil.copy2(repo_template_path, eclipse_template_path)
        template_path = eclipse_template_path
        cstr(f"[SmartLM] Copied template to Eclipse folder for editing: {filename}").msg.print()
    else:
        return web.Response(status=404, text="Template not found")
    
    try:
        # Get updates from request body
        updates = await request.json()
        
        # Read current template
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Apply updates
        changes = []
        for key, value in updates.items():
            if template_data.get(key) != value:
                template_data[key] = value
                changes.append(f"{key}={value}")
        
        # Save if anything changed
        if changes:
            with open(template_path, 'w') as f:
                json.dump(template_data, f, indent=2)
            # Extract template name without .json extension
            template_name = filename.replace('.json', '')
            cstr(f"[SmartLM] ✓ Auto-saved template '{template_name}': {', '.join(changes)}").msg.print()
            return web.json_response({"success": True, "changes": changes})
        else:
            return web.json_response({"success": True, "changes": []})
    
    except Exception as e:
        cstr(f"[SmartLM] Error updating template {filename}: {e}").error.print()
        return web.Response(status=500, text=str(e))

# API route for getting advanced defaults config for Smart LM
@server.PromptServer.instance.routes.get("/eclipse/smartlml_advanced_defaults")
async def get_smartlml_advanced_defaults(request):
    """
    Load advanced defaults config from Eclipse folder first, fall back to repo folder.
    Returns JSON with default parameters for each model type.
    """
    import os
    
    # Check Eclipse folder first (user-editable)
    eclipse_config_path = os.path.join(eclipse_dir, 'config', 'smartlm_advanced_defaults.json')
    
    # Fall back to repo folder if not in Eclipse
    repo_config_path = os.path.join(os.path.dirname(__file__), 'templates', 'config', 'smartlm_advanced_defaults.json')
    
    config_path = eclipse_config_path if os.path.exists(eclipse_config_path) else repo_config_path
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return web.json_response(config_data)
        else:
            # Return empty config if neither exists
            return web.json_response({})
    
    except Exception as e:
        cstr(f"[SmartLM] Error loading advanced defaults: {e}").error.print()
        return web.Response(status=500, text=str(e))

# API route for saving advanced defaults config for Smart LM
@server.PromptServer.instance.routes.post("/eclipse/smartlml_advanced_defaults")
async def post_smartlml_advanced_defaults(request):
    """
    Save advanced defaults config to Eclipse folder.
    Request body should be JSON with model_type and parameters.
    """
    import os
    
    try:
        # Parse request body
        updates = await request.json()
        
        if not updates or 'model_type' not in updates:
            return web.Response(status=400, text="Missing model_type in request")
        
        model_type = updates.pop('model_type')
        params = updates
        
        # Always save to Eclipse folder
        eclipse_config_path = os.path.join(eclipse_dir, 'config', 'smartlm_advanced_defaults.json')
        
        # Ensure Eclipse config directory exists
        os.makedirs(os.path.dirname(eclipse_config_path), exist_ok=True)
        
        # Load current config (from Eclipse or repo)
        repo_config_path = os.path.join(os.path.dirname(__file__), 'templates', 'config', 'smartlm_advanced_defaults.json')
        config_read_path = eclipse_config_path if os.path.exists(eclipse_config_path) else repo_config_path
        
        current_config = {}
        if os.path.exists(config_read_path):
            with open(config_read_path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
        
        # Update parameters for the specific model type
        if model_type not in current_config:
            current_config[model_type] = {}
        
        current_config[model_type].update(params)
        
        # Save to Eclipse config file
        with open(eclipse_config_path, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, indent=2)
        
        # Log changes
        changes = [f"{key}={value}" for key, value in params.items()]
        cstr(f"[SmartLM] ✓ Auto-saved advanced defaults for {model_type}: {', '.join(changes)}").msg.print()
        
        return web.json_response({"success": True, "changes": changes})
    
    except Exception as e:
        cstr(f"[SmartLM] Error saving advanced defaults: {e}").error.print()
        return web.Response(status=500, text=str(e))

# API route for getting folder files for RandomPrompt
@server.PromptServer.instance.routes.get("/eclipse/folder_files/{folder}")
async def get_folder_files(request):
    folder = request.match_info.get('folder', '')
    if not folder:
        return web.json_response({})
    
    import os
    
    # Primary: check wildcards/smartprompt
    folder_path = os.path.join(models_smartprompt_dir, folder)
    if not os.path.isdir(folder_path):
        # Fallback: check repo prompt directory
        folder_path = os.path.join(repo_prompt_dir, folder)
    
    files = {}
    if os.path.isdir(folder_path):
        folder_name = os.path.basename(folder_path)
        clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
        
        # Collect and sort files by number
        folder_files = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith('.txt') and fname.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                try:
                    number = int(fname.split('_')[0])
                    folder_files.append((number, fname))
                except ValueError:
                    continue
        folder_files.sort(key=lambda x: x[0])
        
        for number, fname in folder_files:
            base = os.path.splitext(fname)[0]
            clean_base = re.sub(r'^[0-9_]+', '', base).replace('_', ' ')
            display = f"{clean_folder_name} {clean_base}"
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    files[display] = lines
            except Exception:
                files[display] = []
    
    return web.json_response(files)

# API route for getting widget-to-folder mapping for RandomPrompt
@server.PromptServer.instance.routes.get("/eclipse/widget_folder_mapping")
async def get_widget_folder_mapping(request):
    import os
    import re
    
    # Primary: use wildcards/smartprompt
    prompt_dir = models_smartprompt_dir
    if not os.path.isdir(models_smartprompt_dir):
        # Fallback: use repo prompt directory
        prompt_dir = repo_prompt_dir
    
    mapping = {}
    if os.path.isdir(prompt_dir):
        for item in os.listdir(prompt_dir):
            item_path = os.path.join(prompt_dir, item)
            if os.path.isdir(item_path):
                folder_name = os.path.basename(item_path)
                clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
                
                # Collect and sort files by number
                folder_files = []
                for fname in os.listdir(item_path):
                    if fname.lower().endswith('.txt') and fname.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                        try:
                            number = int(fname.split('_')[0])
                            folder_files.append((number, fname))
                        except ValueError:
                            continue
                folder_files.sort(key=lambda x: x[0])
                
                for number, fname in folder_files:
                    base = os.path.splitext(fname)[0]
                    clean_base = re.sub(r'^[0-9_]+', '', base).replace('_', ' ')
                    display = f"{clean_folder_name} {clean_base}"
                    mapping[display] = clean_folder_name
    
    return web.json_response(mapping)

# Wildcard processor server endpoints will be initialized after nodes are loaded

py = get_ext_dir("py")
files = os.listdir(py)
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]
    imported_module = importlib.import_module(f".py.{name}", __name__)
    try:
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
    except Exception:
        pass

# Initialize wildcard processor server endpoints after loading nodes
# This ensures wildcards are loaded when the endpoints are registered
try:
    from .core.server_endpoints import initialize_endpoints
    initialize_endpoints()
except Exception as e:
    cstr(f"Failed to initialize wildcard processor endpoints: {e}").warning.print()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]