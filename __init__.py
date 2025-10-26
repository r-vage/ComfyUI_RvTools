# Comfyui_RvTools Extension Loader
#
# Initializes and loads all custom nodes for Comfyui_RvTools, providing NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS for the extension.
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
import __main__
from .core import version, cstr
from aiohttp import web 
import server

from typing import Any, Dict, Type

NODE_CLASS_MAPPINGS: Dict[str, Type[Any]] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# MESSAGE TEMPLATES
cstr.color.add_code("msg", f"{cstr.color.LIGHTGREEN}RvTools: {cstr.color.END}")
cstr.color.add_code("warning", f"{cstr.color.LIGHTGREEN}RvTools {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.color.add_code("debug", f"{cstr.color.LIGHTGREEN}RvTools {cstr.color.LIGHTBEIGE}Debug: {cstr.color.END}")
cstr.color.add_code("error", f"{cstr.color.RED}RvTools {cstr.color.END}Error: {cstr.color.END}")

cstr(f'Version: {version}').msg.print()

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)
    dir = os.path.abspath(dir)
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Initialize Smart Prompt files in wildcards folder (one-time copy on first run)
from .core.common import copy_prompt_files_once
comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
repo_prompt_dir = os.path.join(os.path.dirname(__file__), 'prompt')
models_smartprompt_dir = os.path.join(comfyui_root, 'models', 'wildcards', 'smartprompt')
if not os.path.exists(models_smartprompt_dir) and os.path.exists(repo_prompt_dir):
    copy_prompt_files_once(repo_prompt_dir, models_smartprompt_dir)

# API route for serving loader templates
@server.PromptServer.instance.routes.get("/rvtools/loader_templates/{filename}")
async def serve_loader_template(request):
    filename = request.match_info.get('filename', '')
    if not filename.endswith('.json'):
        return web.Response(status=400, text="Invalid file type")
    
    template_dir = get_ext_dir("json/loader_templates")
    template_path = os.path.join(template_dir, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(template_path).startswith(template_dir):
        return web.Response(status=403, text="Access denied")
    
    if os.path.exists(template_path) and os.path.isfile(template_path):
        return web.FileResponse(template_path)
    else:
        return web.Response(status=404, text="Template not found")

# API route for getting template list
@server.PromptServer.instance.routes.get("/rvtools/loader_templates_list")
async def get_loader_templates_list(request):
    from .py.RvLoader_SmartLoader import get_template_list
    templates = get_template_list()
    return web.json_response(templates)

# API route for getting folder files for RandomPrompt
@server.PromptServer.instance.routes.get("/rvtools/folder_files/{folder}")
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
@server.PromptServer.instance.routes.get("/rvtools/widget_folder_mapping")
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