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

import json
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, Tuple, List, cast

from ..core import CATEGORY, cstr

# Some extension must be setting a seed as server-generated seeds were not random. We'll set a new
# seed and use that state going forward.
initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
eclipse_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def new_random_seed():
    # Gets a new random seed from the eclipse_seed_random_state and resetting the previous state.
    global eclipse_seed_random_state
    prev_random_state = random.getstate()
    random.setstate(eclipse_seed_random_state)
    seed = random.randint(0, 2**64 - 1)
    eclipse_seed_random_state = random.getstate()
    random.setstate(prev_random_state)
    return seed


def get_prompt_folders():
    # Get all folders in smart_prompt/ directory. Primary location: models/Eclipse/smart_prompt, fallback: repo/templates/prompt.
    # Get paths
    comfyui_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    eclipse_prompt_dir = os.path.join(comfyui_root, 'models', 'Eclipse', 'smart_prompt')
    repo_prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates', 'prompt')
    
    # Primary: check models/Eclipse/smart_prompt (user-editable location)
    folders = []
    if os.path.isdir(eclipse_prompt_dir):
        for item in os.listdir(eclipse_prompt_dir):
            item_path = os.path.join(eclipse_prompt_dir, item)
            if os.path.isdir(item_path):
                folders.append(item_path)
        if folders:  # If we found folders in Eclipse, use them
            return folders
    
    # Fallback: use repo's prompt directory
    if os.path.isdir(repo_prompt_dir):
        for item in os.listdir(repo_prompt_dir):
            item_path = os.path.join(repo_prompt_dir, item)
            if os.path.isdir(item_path):
                folders.append(item_path)
    
    return folders


class RvText_SmartPrompt_All:
    def __init__(self):
        self.last_seed = None
        self.last_output = None
        self.file_options = None

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "create_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        required: Dict[str, Any] = {}
        
        # Get available folders for the combo box (clean names without numbers)
        prompt_folders = get_prompt_folders()
        folder_names = []
        folder_map = {}  # Map clean name to actual folder path
        for folder in prompt_folders:
            folder_name = os.path.basename(folder)
            # Clean folder name by removing leading numbers and underscores
            clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
            if clean_folder_name not in folder_names:  # Avoid duplicates
                folder_names.append(clean_folder_name)
                folder_map[clean_folder_name] = folder
        
        folder_options = ['All'] + sorted(folder_names)
        required['folder'] = (folder_options, {
            'default': 'subjects',
            'tooltip': 'Select folder to load prompt options from, or All to show all folders'
        })
        
        # Scan all folders and collect widget info
        widget_list = []
        for folder in prompt_folders:
            if not os.path.isdir(folder):
                continue
            folder_name = os.path.basename(folder)
            clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
            
            # Collect files for this folder
            folder_files = []
            for fname in os.listdir(folder):
                if fname.lower().endswith('.txt') and fname.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    try:
                        number = int(fname.split('_')[0])
                        folder_files.append((number, fname))
                    except ValueError:
                        continue  # Skip files that don't start with number_
            
            # Sort files by number
            folder_files.sort(key=lambda x: x[0])
            
            for number, fname in folder_files:
                base = os.path.splitext(fname)[0]
                # Clean widget name by removing all leading numbers and underscores, and replacing remaining underscores with spaces
                clean_base = re.sub(r'^[0-9_]+', '', base).replace('_', ' ')
                display = f"{clean_folder_name} {clean_base}"
                fpath = os.path.join(folder, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        combo_options = ['None', 'Random'] + lines
                        tooltip_dict = {"default": "None", "tooltip": f"Select entry from {fname} in {clean_folder_name}", "folder": clean_folder_name}
                except Exception:
                    combo_options = ['None']
                    tooltip_dict = {"default": "None", "tooltip": f"Select entry from {fname} in {clean_folder_name}", "folder": clean_folder_name}
                widget_list.append((display, combo_options, tooltip_dict))
        
        # No global sort needed, widgets are already in folder order with internal number sorting
        
        # Add sorted widgets to required
        for display, combo_options, tooltip_dict in widget_list:
            required[display] = (combo_options, tooltip_dict)
        
        # Add seed as the last parameter
        required["seed"] = ("INT", {"default": 0, "min": -3, "max": 2**64 - 1, "tooltip": "Random seed for prompt selection."})
        
        return {
            "required": required,
            "optional": {
                "seed_input": ("INT", {"default": None, "forceInput": True, "tooltip": "Optional seed input that overrides the widget seed if connected"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        # Forces a changed state if we happen to get a special seed, as if from the API directly.
        if seed in (-1, -2, -3):
            # This isn't used, but a different value than previous will force it to be "changed"
            return new_random_seed()
        
        folder = kwargs.get('folder', 'All')
        # Also check if any widget values have changed
        # Create a hash of all widget values that affect the output
        widget_values = tuple(sorted(kwargs.items()))
        return (seed, folder, widget_values)

    def create_prompt(self, seed=0, seed_input=None, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        # Use seed_input if provided (which will be the actual executed seed from connected node)
        # Otherwise use the widget seed
        original_seed = seed
        if seed_input is not None:
            # seed_input contains the actual executed seed value from the connected node
            seed = seed_input
            original_seed = seed_input
            
        # Handle special seeds (-1, -2, -3) only if NOT from seed_input
        # (seed_input will already have resolved seeds from the connected node)
        if seed_input is None and seed in (-1, -2, -3):
            if seed in (-2, -3):
                cstr(f'Cannot {"increment" if seed == -2 else "decrement"} seed from ' +
                     'server, but will generate a new random seed.').warning.print()

            seed = new_random_seed()
            cstr(f'Server-generated random seed {seed} used for random prompt selection.').msg.print()

            # Save the resolved seed to workflow
            if unique_id is not None and extra_pnginfo is not None:
                workflow_node = next(
                    (x for x in extra_pnginfo['workflow']['nodes'] if str(x['id']) == str(unique_id)), None)
                if workflow_node is not None and 'widgets_values' in workflow_node:
                    for index, widget_value in enumerate(workflow_node['widgets_values']):
                        if widget_value == original_seed:
                            workflow_node['widgets_values'][index] = seed
                            break

            if prompt is not None:
                prompt_node = prompt.get(str(unique_id))
                if prompt_node is not None and 'inputs' in prompt_node and 'seed' in prompt_node['inputs']:
                    prompt_node['inputs']['seed'] = seed

        # Get selected folder (clean name)
        selected_folder = kwargs.get('folder', 'All')
        
        # Build prompt from selected or random lines
        # Create a cache key that includes seed, folder, and widget values
        widget_values = tuple(sorted(kwargs.items()))
        cache_key = (seed, selected_folder, widget_values)
        
        if self.last_seed == seed and self.last_output is not None and getattr(self, 'last_folder', None) == selected_folder and getattr(self, 'last_widget_values', None) == widget_values:
            return (self.last_output,)
        
        # Store current values for caching
        self.last_widget_values = widget_values
        self.last_folder = selected_folder
        
        # Build file map only for selected folder(s)
        file_map = {}
        prompt_folders = get_prompt_folders()
        
        folders_to_scan = []
        if selected_folder == 'All':
            folders_to_scan = prompt_folders
        else:
            # Find folders that match the clean name
            for folder in prompt_folders:
                folder_name = os.path.basename(folder)
                clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
                if clean_folder_name == selected_folder:
                    folders_to_scan.append(folder)
        
        for folder in folders_to_scan:
            if not os.path.isdir(folder):
                continue
            folder_name = os.path.basename(folder)
            clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
            
            # Collect files for this folder
            folder_files = []
            for fname in os.listdir(folder):
                if fname.lower().endswith('.txt') and fname.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    try:
                        number = int(fname.split('_')[0])
                        folder_files.append((number, fname))
                    except ValueError:
                        continue
            
            # Sort files by number
            folder_files.sort(key=lambda x: x[0])
            
            for number, fname in folder_files:
                base = os.path.splitext(fname)[0]
                # Use clean widget name (same as in INPUT_TYPES)
                clean_base = re.sub(r'^[0-9_]+', '', base).replace('_', ' ')
                display = f"{clean_folder_name} {clean_base}"
                fpath = os.path.join(folder, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        file_map[display] = lines
                except Exception:
                    file_map[display] = []
        
        values = []
        random.seed(seed)
        random_selections = {}  # Track which widgets had "Random" and their selected values
        
        for display, lines in file_map.items():
            val = kwargs.get(display, "None")
            if val == "Random":
                if lines:
                    selected = random.choice(lines)
                    values.append(selected)
                    random_selections[display] = selected  # Store the resolved value
            elif val not in ("None", "disabled"):
                values.append(val.strip())
        
        # Save resolved random values to workflow metadata (same pattern as seed resolution)
        if random_selections and unique_id is not None and extra_pnginfo is not None:
            workflow_node = next(
                (x for x in extra_pnginfo['workflow']['nodes'] if str(x['id']) == str(unique_id)), None)
            if workflow_node is not None and 'widgets_values' in workflow_node:
                # Rebuild the widget list in the same order as INPUT_TYPES to find correct indices
                widget_order = ['folder']  # First widget is always 'folder'
                
                # Rebuild widgets in the same order as INPUT_TYPES
                prompt_folders = get_prompt_folders()
                for folder in prompt_folders:
                    if not os.path.isdir(folder):
                        continue
                    folder_name = os.path.basename(folder)
                    clean_folder_name = re.sub(r'^[0-9_]+', '', folder_name)
                    
                    folder_files = []
                    for fname in os.listdir(folder):
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
                        widget_order.append(display)
                
                widget_order.append('seed')  # Seed is the last widget
                
                # Update widgets_values with resolved random selections
                for widget_name, selected_value in random_selections.items():
                    if widget_name in widget_order:
                        index = widget_order.index(widget_name)
                        if index < len(workflow_node['widgets_values']):
                            workflow_node['widgets_values'][index] = selected_value
        
        # Also update the prompt inputs for consistency
        if random_selections and prompt is not None:
            prompt_node = prompt.get(str(unique_id))
            if prompt_node is not None and 'inputs' in prompt_node:
                for widget_name, selected_value in random_selections.items():
                    if widget_name in prompt_node['inputs']:
                        prompt_node['inputs'][widget_name] = selected_value
        
        # Clean up values: remove trailing punctuation and extra spaces
        values = [re.sub(r'[.,;:!?]+$', '', val.strip()) for val in values]
        
        # Join with comma and space
        prompt = ', '.join(values)
        
        # Clean up the final prompt: multiple spaces to single, remove trailing comma
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        prompt = re.sub(r',\s*$', '', prompt)
        
        self.last_seed = seed
        self.last_output = prompt
        return (prompt,)

NODE_NAME = 'Smart Prompt [Eclipse]'
NODE_DESC = 'Smart Prompt'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvText_SmartPrompt_All
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
