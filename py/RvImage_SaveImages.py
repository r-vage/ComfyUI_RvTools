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

import os
import re
import torch
import json
import numpy as np
import folder_paths
import hashlib

from pathlib import Path
from typing import Optional, Final, Dict, List, Union, Any
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..core import CATEGORY, cstr, purge_vram

UPSCALE_MODELS = folder_paths.get_filename_list("upscale_models") + ["None"]
MAX_RESOLUTION = 32768

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

# Global variables to store values
global_values = {
    'model': '',
    'basemodel': '',
    'seed': '',
    'sampler_name': '',
    'scheduler': '',
    'steps': '',
    'cfg': '',
    'denoise': '',
    'clip_skip': ''
}

from datetime import datetime

class FilenameProcessor:
#     Handles filename placeholder processing with improved error handling and type safety
    
    def __init__(self):
        self.placeholders = {
            '%today': self._get_date,
            '%date': self._get_date,
            '%time': self._get_time,
            # Individual date/time components for custom formatting
            '%Y': lambda: datetime.now().strftime('%Y'),
            '%m': lambda: datetime.now().strftime('%m'),
            '%M': lambda: datetime.now().strftime('%m'),
            '%d': lambda: datetime.now().strftime('%d'),
            '%D': lambda: datetime.now().strftime('%d'),
            '%H': lambda: datetime.now().strftime('%H'),
            '%S': lambda: datetime.now().strftime('%S'),
            '%basemodel': lambda: str(global_values.get('basemodel', '')),
            '%model': lambda: str(global_values.get('model', '')),
            '%seed': lambda: str(global_values.get('seed', '')),
            '%sampler_name': lambda: str(global_values.get('sampler_name', '')),
            '%scheduler': lambda: str(global_values.get('scheduler', '')),
            '%steps': lambda: str(global_values.get('steps', '')),
            '%cfg': lambda: str(global_values.get('cfg', '')),
            '%denoise': lambda: str(global_values.get('denoise', '')),
            '%clip_skip': lambda: str(global_values.get('clip_skip', ''))
        }

    @staticmethod
    def _get_date() -> str:
#         Get current date in YYYY-MM-DD format
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _get_time() -> str:
#         Get current time in HHMMSS format
        return datetime.now().strftime("%H%M%S")

    def get_used_placeholders(self, filename: str) -> List[str]:
        # Get list of placeholders used in filename
        if not isinstance(filename, str):
            cstr(f"Invalid filename type: {type(filename)}").warning.print()
            return []
        
        return [p for p in self.placeholders.keys() if p in filename]

    def get_placeholder_value(self, placeholder: str) -> str:
        # Get value for a specific placeholder
        try:
            if placeholder not in self.placeholders:
                # If we don't have a value for this placeholder (for example
                # because no pipe is connected), return the placeholder name
                # without the leading '%' so filenames remain readable.
                cstr(f"Unknown placeholder: {placeholder}; falling back to name without %").debug.print()
                return placeholder.lstrip('%')

            value = self.placeholders[placeholder]()

            # If the resolved value is falsy (None or empty), fall back to
            # the placeholder name without '%' so the placeholder doesn't
            # remain literally in the filename.
            if value in (None, ''):
                cstr(f"Placeholder {placeholder} resolved to empty; falling back to name without %").debug.print()
                return placeholder.lstrip('%')

            return str(value)
            
        except Exception as e:
            cstr(f"Error getting value for {placeholder}: {e}").error.print()
            return ''

    def process_string(self, filename_prefix: str, isPath: bool) -> str:
        # Process filename replacing all placeholders with their values
        try:
            if not filename_prefix or not isinstance(filename_prefix, str):
                cstr("Invalid filename_prefix").warning.print()
                return "default"

            # Get all placeholders used in this filename
            used_placeholders = self.get_used_placeholders(filename_prefix)
            if not used_placeholders:
                return filename_prefix

            # Replace each placeholder
            result = filename_prefix
            for placeholder in used_placeholders:
                value = self.get_placeholder_value(placeholder)
                result = result.replace(placeholder, value)

            # Sanitize final filename
            if isPath:
                return self.sanitize_path(result)
            else:
                return self.sanitize_filename(result)

        except Exception as e:
            cstr(f"Error processing filename: {e}").error.print()
            return "error_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        # Remove invalid characters from filename for both Windows and Linux
        windows_invalid = '<>:"/\\|?*'
        linux_invalid = '/'
        control_chars = ''.join(chr(i) for i in range(32))  # ASCII control characters

        # Replace invalid characters
        for char in windows_invalid + linux_invalid + control_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots (problematic in Windows)
        filename = filename.strip(' .')
        
        # Ensure filename isn't empty and has reasonable length
        if not filename:
            return "untitled"
            
        # Handle Windows reserved names
        windows_reserved = {
            'CON', 'PRN', 'AUX', 'NUL', 
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        name_without_ext = filename.split('.')[0].upper()
        if name_without_ext in windows_reserved:
            filename = '_' + filename
            
        # Truncate if too long (Windows MAX_PATH limitation)
        if len(filename) > 255:
            base, ext = os.path.splitext(filename)
            filename = base[:255-len(ext)] + ext
            
        return filename

    @staticmethod
    def sanitize_path(path: str) -> str:
        # Remove invalid characters from path for both Windows and Linux
        parts = Path(path).parts
        
        # Sanitize each component
        sanitized_parts = []
        for i, part in enumerate(parts):
            if i == 0 and len(parts) > 1 and part.endswith(':'):
                # Handle Windows drive letter (e.g., C:)
                sanitized_parts.append(part)
            else:
                # Define invalid characters for path components
                windows_invalid = '<>:"|?*'  # Note: removed / and \ as they're path separators
                linux_invalid = ''  # Linux allows most characters in paths except /
                control_chars = ''.join(chr(i) for i in range(32))
                
                # Replace invalid characters
                for char in windows_invalid + linux_invalid + control_chars:
                    part = part.replace(char, '_')
                
                # Remove leading/trailing spaces and dots
                part = part.strip(' .')
                
                # Ensure part isn't empty
                if not part:
                    part = "unnamed"
                    
                sanitized_parts.append(part)
        
        # Reconstruct path
        sanitized_path = str(Path(*sanitized_parts))
        
        # Ensure path isn't too long
        if len(sanitized_path) > 255:
            cstr(f"Path too long, may cause issues on some systems: {sanitized_path}").warning.print()
            
        return sanitized_path

def set_global_values(
    model: Optional[str] = None,
    basemodel: Optional[str] = None,
    seed_value: Optional[Union[int, float]] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    steps: Optional[Union[int, float]] = None,
    cfg: Optional[Union[int, float]] = None,
    denoise: Optional[Union[int, float]] = None,
    clip_skip: Optional[Union[int, float]] = None
) -> None:

    # Safely set global values with improved type checking and validation

    try:
        value_types = {
            'model': str, 
            'basemodel': str, 
            'seed': (int, float),
            'sampler_name': str,
            'scheduler': str, 
            'steps': (int, float),
            'cfg': (int, float),
            'denoise': (int, float),
            'clip_skip': (int, float)
        }

        values = {
            'model': model,
            'basemodel': basemodel,
            'seed': seed_value,
            'sampler_name': sampler_name,
            'scheduler': scheduler,
            'steps': steps,
            'cfg': cfg,
            'denoise': denoise,
            'clip_skip': clip_skip
        }

        # Process each value with strict type checking
        for key, value in values.items():
            # Skip None values - leave as default empty string
            if value is None:
                continue
                
            # Treat explicit empty strings and 'None' as unset for numeric fields
            if value in ('', 'None'):
                # leave as default (empty string)
                continue
            
            expected_type = value_types[key]
            
            # Validate type
            if not isinstance(value, expected_type):  # type: ignore[arg-type]
                try:
                    # Try type conversion for numbers
                    if expected_type in [(int, float), float]:
                        value = float(value)
                    elif expected_type == int:
                        value = int(value)
                    elif expected_type == str:
                        value = str(value)
                except (ValueError, TypeError) as e:
                    cstr(f"Ignoring non-numeric/invalid value for {key}: {e}").debug.print()
                    # leave as default empty to avoid noisy errors
                    value = ''

            # Additional validation
            if isinstance(value, (int, float)):
                # Ensure numeric values are reasonable
                if key in ['steps', 'cfg', 'denoise']:
                    if value < 0:
                        cstr(f"Negative value for {key} adjusted to 0").warning.print()
                        value = 0
            
            global_values[key] = str(value)

    except Exception as e:
        cstr(f"Error in set_global_values: {e}").error.print()
        # Reset to safe defaults
        for key in value_types.keys():
            global_values[key] = ''

# Initialize the filename processor as a singleton
filename_processor = FilenameProcessor()

def string_placeholder(filename_prefix: str, isPath: bool) -> str:
    # Public interface for filename processing
    return filename_processor.process_string(filename_prefix, isPath)


# Constants for configuration
CHUNK_SIZE: Final[int] = 8192  # Optimal chunk size for reading files
MAX_WORKERS: Final[int] = 4    # Number of concurrent hash operations
HASH_CACHE: Dict[str, str] = {}  # Cache for hash values



def get_sha256(file_path: str) -> Optional[str]:
    # Calculate or retrieve SHA-256 hash for a file with improved safety and caching.
    if not file_path or file_path in ('undefined', 'none'):
        cstr(f"Invalid file path: {file_path}").warning.print()
        return None
    try:
        file_path = str(Path(file_path).resolve())
        cache_key = f"sha256:{file_path}"
        if cache_key in HASH_CACHE:
            return HASH_CACHE[cache_key]
        file_no_ext = str(Path(file_path).with_suffix(''))
        hash_file = file_no_ext + ".sha256"
        try:
            if Path(hash_file).exists():
                with open(hash_file, "r") as f:
                    hash_value = f.read().strip()
                    if len(hash_value) == 64:
                        HASH_CACHE[cache_key] = hash_value
                        return hash_value
        except OSError as e:
            cstr(f"Error reading hash file {hash_file}: {e}").error.print()
        if not Path(file_path).exists():
            cstr(f"Source file not found: {file_path}").error.print()
            return None
        cstr(f"Calculating SHA-256 for: {Path(file_path).name}").msg.print()
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                hash_obj.update(chunk)
        hash_value = hash_obj.hexdigest()
        HASH_CACHE[cache_key] = hash_value
        try:
            with open(hash_file, "w") as f:
                f.write(hash_value)
        except OSError as e:
            cstr(f"Failed to save hash file {hash_file}: {e}").error.print()
        return hash_value
    except Exception as e:
        cstr(f"Hash calculation failed for {file_path}: {e}").error.print()
        return None


# Represent the given embedding name as key as detected by civitAI

def civitai_embedding_key_name(embedding: str):
    return f'embed:{embedding}'

# Represent the given lora name as key as detected by civitAI
# NB: this should also work fine for Lycoris

def civitai_lora_key_name(lora: str):
    return f'LORA:{lora}'

def civitai_model_key_name(model: str):
    return f'Model:{model}'

# Based on a embedding name, eg: EasyNegative, finds the path as known in comfy, including extension

def full_embedding_path_for(embedding: str):
    # Match by filename (without extension) in a case-insensitive manner
    name = str(embedding)
    matching_embedding = None
    # Try direct match in embeddings folder
    for x in __list_embeddings():
        if Path(x).name.lower().startswith(name.lower()):
            matching_embedding = x
            break
    # If not found, try subfolder or direct path
    if not matching_embedding:
        if os.sep in name or '/' in name:
            candidate = os.path.join(folder_paths.get_folder_paths("embeddings")[0], name)
            if os.path.exists(candidate):
                matching_embedding = name
    # Try with supported extensions
    if not matching_embedding:
        for ext in ['.pt', '.safetensors', '.bin']:
            candidate = name if name.lower().endswith(ext) else name + ext
            candidate_path = os.path.join(folder_paths.get_folder_paths("embeddings")[0], candidate)
            if os.path.exists(candidate_path):
                matching_embedding = candidate
                break
    if not matching_embedding:
        return None
    return folder_paths.get_full_path("embeddings", matching_embedding)

# Based on a lora name, e.g., 'epi_noise_offset2', finds the path as known in comfy, including extension.

def full_lora_path_for(lora: str):
    # Strip any weight or trailing tokens (e.g. 'name:0.8' or '<lora:name:0.8>')
    # and strip extension if present. We match by filename (without extension)
    # to ensure hashes and civitai keys use the base filename only.
    original = str(lora)
    # If tokenized form <lora:name:weight>, extract the name
    m = re.search(r'<lora:([^>:]+)', original)
    if m:
        name = m.group(1)
    else:
        # If contains a weight like name:0.8, split it off
        name = original.split(':')[0]

    matching_lora = None
    # Try direct match in loras folder
    for x in __list_loras():
        if Path(x).name.lower().startswith(name.lower()):
            matching_lora = x
            break
    # If not found, try subfolder or direct path
    if not matching_lora:
        if os.sep in name or '/' in name:
            candidate = os.path.join(folder_paths.get_folder_paths("loras")[0], name)
            if os.path.exists(candidate):
                matching_lora = name
    # Try with supported extensions
    if not matching_lora:
        for ext in ['.safetensors', '.pt', '.bin']:
            candidate = name if name.lower().endswith(ext) else name + ext
            candidate_path = os.path.join(folder_paths.get_folder_paths("loras")[0], candidate)
            if os.path.exists(candidate_path):
                matching_lora = candidate
                break
    if not matching_lora:
        cstr(f'Eclipse: could not find full path to lora "{original}"').error.print()
        return None
    return folder_paths.get_full_path("loras", matching_lora)


def parse_lora_string(lora_input):
    # Parse various lora input formats into a normalized token string and a dict of weights.
    # Supported formats:
    # - '<lora:name:weight><lora:name2:weight>' (already tokenized)
    # - 'name:weight,name2' (comma-separated, optional weights)
    # - 'name name2' (space-separated)
    # Returns: (token_string, {name: weight, ...})
    # Deduplicates loras, keeping the first occurrence

    import re
    tokens = []
    weights = {}
    seen_loras = set()  # Track seen lora names (normalized)

    if lora_input is None:
        return ('', {})

    # If the input already contains <lora:...> tokens, extract them
    token_matches = re.findall(r'<lora:([^>:]+):?([0-9\.]+)?[^>]*>', str(lora_input))
    if token_matches:
        for name, w in token_matches:
            name = Path(name).stem  # Remove path and extension
            name_normalized = name.lower()
            # Skip if we've already seen this lora
            if name_normalized in seen_loras:
                continue
            seen_loras.add(name_normalized)
            weight = float(w) if w not in (None, '') else 1.0
            tokens.append(f"<lora:{name}:{weight}>")
            weights[name] = weight
        return (''.join(tokens), weights)

    # Otherwise, accept comma or whitespace separated lists like 'a:0.8,b' or 'a b'
    parts = [p.strip() for p in re.split(r'[,;\s]+', str(lora_input)) if p.strip()]
    for part in parts:
        if ':' in part:
            name, w = part.split(':', 1)
            name = Path(name).stem  # Remove path and extension
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
        else:
            name = Path(part).stem  # Remove path and extension
            weight = 1.0
        
        name_normalized = name.lower()
        # Skip if we've already seen this lora
        if name_normalized in seen_loras:
            continue
        seen_loras.add(name_normalized)
        
        tokens.append(f"<lora:{name}:{weight}>")
        weights[name] = weight

    return (''.join(tokens), weights)


def __list_loras():
    return folder_paths.get_filename_list("loras")

def __list_embeddings():
    return folder_paths.get_filename_list("embeddings")

# Extracts Embeddings and Lora's from the given prompts
# and allows asking for their sha's 
# This module is based on civit's plugin and website implementations
# The image saver node goes through the automatic flow, not comfy, on civit
# see: https://github.com/civitai/sd_civitai_extension/blob/2008ba9126ddbb448f23267029b07e4610dffc15/scripts/gen_hashing.py
# see: https://github.com/civitai/civitai/blob/d83262f401fb372c375e6222d8c2413fa221c2c5/src/utils/metadata/automatic.metadata

class PromptMetadataExtractor:
    # Anything that follows embedding:<characters except , or whitespace
    EMBEDDING = r'embedding:([^,\s\(\)\:]+)'
    # Anything that follows <lora:NAME> with allowance for :weight, :weight.fractal or LBW
    # Match tokenized <lora:name:weight> or any <lora:name...> and capture the base name
    LORA = r'<lora:([^>:]+)(?::[^>]+)?>'

    def __init__(self, prompts: List[str]):
        self.__embeddings: dict[str, Optional[str]] = {}
        self.__loras: dict[str, Optional[str]] = {}
        self.__perform(prompts)

    # Returns the embeddings used in the given prompts in a format as known by civitAI
    # Example output: {"embed:EasyNegative": "66a7279a88", "embed:FastNegativeEmbedding": "687b669d82", "embed:ng_deepnegative_v1_75t": "54e7e4826d", "embed:imageSharpener": "fe5a4dfc4a"}
    
    def get_embeddings(self):
        return self.__embeddings
    
    # Returns the lora's used in the given prompts in a format as known by civitAI
    # Example output: {"LORA:epi_noiseoffset2": "81680c064e", "LORA:GoodHands-beta2": "ba43b0efee"}

    def get_loras(self):
        return self.__loras

    # Private API
    def __perform(self, prompts):
        for prompt in prompts:
            embeddings = re.findall(self.EMBEDDING, prompt, re.IGNORECASE | re.MULTILINE)
            for embedding in embeddings:
                self.__extract_embedding_information(embedding)
            
            # Find lora tokens; they may include weights. We only keep base name
            lora_matches = re.findall(self.LORA, prompt, re.IGNORECASE | re.MULTILINE)
            for lora in lora_matches:
                base = lora.split(':')[0] if ':' in lora else lora
                self.__extract_lora_information(base)

    def __extract_embedding_information(self, embedding: str):
        embedding_name = civitai_embedding_key_name(embedding)
        embedding_path = full_embedding_path_for(embedding)
        if embedding_path is None or not os.path.exists(embedding_path):
            cstr(f"Embedding file not found for hash: {embedding}").warning.print()
            return
        sha_full = get_sha256(embedding_path)
        if sha_full:
            sha = sha_full[:10]
        else:
            sha = None
        self.__embeddings[embedding_name] = sha

    def __extract_lora_information(self, lora: str):
        # Get the full path first
        lora_path = full_lora_path_for(lora)
        if lora_path is None or not os.path.exists(lora_path):
            cstr(f"Lora file not found for hash: {lora}").warning.print()
            return
        
        # Extract filename without extension, preserving dots in the name
        # e.g., "flux\standard\FLUX.1-Turbo-Alpha.safetensors" -> "FLUX.1-Turbo-Alpha"
        lora_filename = os.path.basename(lora_path)
        # Remove extension manually to preserve dots in filename
        if lora_filename.lower().endswith('.safetensors'):
            lora_base = lora_filename[:-12]  # Remove .safetensors
        elif lora_filename.lower().endswith('.pt'):
            lora_base = lora_filename[:-3]  # Remove .pt
        elif lora_filename.lower().endswith('.bin'):
            lora_base = lora_filename[:-4]  # Remove .bin
        else:
            # Fallback to splitext
            lora_base = os.path.splitext(lora_filename)[0]
        
        lora_name = civitai_lora_key_name(lora_base)
        sha_full = get_sha256(lora_path)
        if sha_full:
            sha = sha_full[:10]
        else:
            sha = None
        self.__loras[lora_name] = sha
    
    def __get_shortened_sha(self, file_path: str):
        sha = get_sha256(file_path)
        return sha[:10] if sha else None


def return_filename(ckpt_name):
    return os.path.basename(ckpt_name)

def return_filename_without_extension(ckpt_name):
    return os.path.splitext(return_filename(ckpt_name))[0]

def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

def save_json(image_info, filename):
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            cstr(f"No image info found, skipping saving of JSON").warning.print()
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            cstr(f"Workflow saved to: '{filename}.json'").msg.print()
    except Exception as e:
        cstr(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution').error.print()


class RvImage_SaveImages:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.civitai_sampler_map = {
            'euler_ancestral': 'Euler a',
            'euler': 'Euler',
            'lms': 'LMS',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_3m_sde': 'DPM++ 3M SDE',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc_bh2': 'UniPC',
            'uni_pc': 'UniPC',
            'lcm': 'LCM',
        }
        self.type = 'output'


    def get_civitai_sampler_name(self, sampler_name, scheduler):
        # based on: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts#L122
        if sampler_name in self.civitai_sampler_map:
            civitai_name = self.civitai_sampler_map[sampler_name]

            if scheduler == "karras":
                civitai_name += " Karras"
            elif scheduler == "exponential":
                civitai_name += " Exponential"
            elif scheduler == "sgm_uniform":
                civitai_name += " SGM Uniform"
            elif scheduler == "simple":
                civitai_name += " Simple"
            elif scheduler == "ddim_uniform":
                civitai_name += " DDIM Uniform"
            elif scheduler == "beta":
                civitai_name += " Beta"
            elif scheduler == "linear_quadratic":
                civitai_name += " Linear Quadratic"
            elif scheduler == "kl_optimal":
                civitai_name += " kl optimal"    
            elif scheduler == "AYS SDXL":
                civitai_name += " AYS SDXL"
            elif scheduler == "AYS SD1":
                civitai_name += " AYS SD1"
            elif scheduler == "AYS SVD":
                civitai_name += " AYS SVD"
            elif scheduler == "simple_test":
                civitai_name += " Simple Test"

            return civitai_name
        else:
            if scheduler != 'normal':
                return f"{sampler_name}_{scheduler}"
            else:
                return sampler_name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # images may be omitted when a pipe is provided (metadata-only save)
                "output_path": ("STRING", {"default": r'%Y-%M-%D\%basemodel', "multiline": False}),
                # Use a raw string for the default output_path to avoid invalid escape sequence warnings
                "output_path": ("STRING", {"default": r'%Y-%M-%D\%basemodel', "multiline": False}),
                "filename_prefix": ("STRING", {"default": "%today, %time, %basemodel, %seed, %sampler_name, %scheduler, %steps, %cfg, %denoise"}),
                "filename_delimiter": ("STRING", {"default":"_"}),
                "filename_number_padding": ("INT", {"default":4, "min":1, "max":9, "step":1}),
                "filename_number_start": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "extension": (['png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'bmp'], ),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "optimize_image": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "lossless_webp": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "embed_workflow": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "save_generation_data": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "remove_prompts": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "add_loras_to_prompt": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "Purge_VRAM": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "show_previews": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
            },
            "optional": {
                "images": ("IMAGE", ),
                "pipe_opt": ("pipe",),
            },

            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.IMAGE.value
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "files",)

    FUNCTION = "save_images"

    OUTPUT_NODE = True

    def save_images(self, 
                        images=None, 
                        output_path='', 
                        filename_prefix="image", 
                        filename_delimiter='_', 
                        filename_number_padding=4, 
                        filename_number_start=False, 
                        extension='png', 
                        dpi=300, 
                        quality=100, 
                        optimize_image=False, 
                        lossless_webp=True, 
                        embed_workflow=True, 
                        save_generation_data=False,
                        remove_prompts=False,
                        save_workflow_as_json=False, 
                        add_loras_to_prompt=False,
                        Purge_VRAM=False,
                        show_previews=False, 
                        enabled=True,
                        pipe_opt=None,
                        prompt=None, 
                        extra_pnginfo=None
                        ):

        # If disabled, just pass through the images without saving
        if not enabled:
            # Determine images to return
            if images is not None:
                return_images = images
            elif pipe_opt is not None:
                # Extract images from pipe
                if isinstance(pipe_opt, tuple) and len(pipe_opt) > 0:
                    ctx = pipe_opt[0] if isinstance(pipe_opt[0], dict) else {}
                elif isinstance(pipe_opt, dict):
                    ctx = pipe_opt
                else:
                    ctx = {}
                pipe_images = ctx.get("images")
                if pipe_images is not None:
                    return_images = pipe_images
                else:
                    return_images = None
            else:
                return_images = None
            return {"ui": {"images": []}, "result": (return_images, [])}

        # Require either images or a pipe (containing metadata) to proceed
        if images is None and pipe_opt is None:
            raise RuntimeError("RvImage_SaveImages requires either an image input or a pipe input (pipe_opt).")

        
        if pipe_opt is not None:
            # Handle both dict and tuple (from context nodes) pipes
            if isinstance(pipe_opt, tuple) and len(pipe_opt) > 0:
                ctx = pipe_opt[0] if isinstance(pipe_opt[0], dict) else {}
            elif isinstance(pipe_opt, dict):
                ctx = pipe_opt
            else:
                raise ValueError("RvImage_SaveImages expects dict-style or tuple-style pipes for pipe_opt.")
            sampler_name = ctx.get("sampler_name") 
            scheduler = ctx.get("scheduler")
            steps = ctx.get("steps")
            cfg = ctx.get("cfg")
            seed_value = ctx.get("seed") 
            width = ctx.get("width")
            height = ctx.get("height")
            positive = ctx.get("text_pos")
            negative = ctx.get("text_neg")
            model_name = ctx.get("model_name") 
            vae_name = ctx.get("vae_name") 
            lora_names = ctx.get("lora_names") or ctx.get("loras")
            denoise = ctx.get("denoise") if ctx.get("denoise") is not None else ctx.get("guidance")
            clip_skip = ctx.get("clip_skip")

            # Try to set global values if we have sampler/seed info
            try:
                set_global_values('', '', seed_value, sampler_name, scheduler, steps, cfg, denoise, clip_skip)
            except Exception as e:
                cstr(f"Failed to set global values: {e}").error.print()

            # Normalize prompt fields to strings. Some graphs may supply lists of
            # tokens or nested lists (e.g. tokenized prompts). Ensure we always
            # work with a plain string for concatenation and metadata extraction.
            def _prompt_to_str(x):
                if x is None:
                    return ""
                # Already a string-like object
                if isinstance(x, (str, bytes)):
                    return str(x)
                # Flatten nested iterables (lists/tuples) into space-separated strings
                if isinstance(x, (list, tuple)):
                    parts = []
                    for item in x:
                        # recurse for nested lists
                        if isinstance(item, (list, tuple)):
                            part = _prompt_to_str(item)
                        else:
                            part = str(item)
                        if part:
                            parts.append(part)
                    return " ".join(parts)
                # Fallback: stringify any other object
                try:
                    return str(x)
                except Exception:
                    return ""

            # Apply normalization, treating common empty markers as empty strings
            positive = _prompt_to_str(positive) if positive not in (None, '', 'undefined', 'none') else ""
            negative = _prompt_to_str(negative) if negative not in (None, '', 'undefined', 'none') else ""

            model_string = {}
            modelhash: Optional[str] = ""
            vae_hash: Optional[str] = ""

            if not model_name in (None, '', 'undefined', 'none'):
                if model_name is not None:
                    # Split and deduplicate models while preserving order
                    models = model_name.split(', ')
                    seen = set()
                    unique_models = []
                    for model in models:
                        model_stripped = model.strip()
                        # Use normalized name (without extension) for dedup comparison
                        model_normalized = return_filename_without_extension(model_stripped).lower()
                        if model_normalized and model_normalized not in seen:
                            seen.add(model_normalized)
                            unique_models.append(model_stripped)
                    
                    models = unique_models
                    
                    # Get first model for basemodel
                    if models and models[0]:
                        first_model = models[0].strip()
                        global_values['basemodel'] = return_filename_without_extension(first_model)
                        global_values['model'] = first_model

                import glob

                def find_model_file(model, search_dirs, extensions):
                    for search_dir in search_dirs:
                        for ext in extensions:
                            pattern = os.path.join(search_dir, '**', model if model.lower().endswith(ext) else model + ext)
                            matches = glob.glob(pattern, recursive=True)
                            if matches:
                                return matches[0], search_dir  # Return both path and directory
                    if os.path.exists(model):
                        return model, None
                    for ext in extensions:
                        candidate = model if model.lower().endswith(ext) else model + ext
                        if os.path.exists(candidate):
                            return candidate, None
                    return None, None

                search_dirs = []
                for key in ["checkpoints", "diffusion_models", "unet", "upscale_models"]:
                    search_dirs.extend(folder_paths.get_folder_paths(key))
                extensions = ['.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.gguf']

                # Get upscale model directories for accurate detection
                upscale_model_dirs = set(folder_paths.get_folder_paths("upscale_models"))

                for model in models:
                    if not model in (None, '', 'undefined', 'none'):
                        model_path, model_dir = find_model_file(model, search_dirs, extensions)
                        modelhash = None
                        if model_path and os.path.exists(model_path):
                            modelhash = get_sha256(model_path)
                            if modelhash:
                                modelhash = modelhash[:10]
                        else:
                            cstr(f"Model file not found for hash: {model} (path: {model_path}, dir: {model_dir})").warning.print()
                        if modelhash not in (None, '', 'undefined', 'none'):
                            # Determine key format based on model type
                            if model_dir and model_dir in upscale_model_dirs:
                                # Upscale models use filename without extension as key
                                model_key = return_filename_without_extension(model)
                                cstr(f"Processing upscale model: {model} -> key: {model_key}, hash: {modelhash}").debug.print()
                            else:
                                # Checkpoints/diffusion models use Civitai format
                                model_key = civitai_model_key_name(return_filename_without_extension(model))
                                cstr(f"Processing checkpoint model: {model} -> key: {model_key}, hash: {modelhash}").debug.print()
                            model_string[model_key] = modelhash

            if not vae_name in (None, '', 'undefined', 'none'):
                if vae_name is not None:
                    # Split and deduplicate VAE models while preserving order
                    models = vae_name.split(', ')
                    seen = set()
                    unique_models = []
                    for model in models:
                        model_stripped = model.strip()
                        model_normalized = return_filename_without_extension(model_stripped).lower()
                        if model_normalized and model_normalized not in seen:
                            seen.add(model_normalized)
                            unique_models.append(model_stripped)
                    
                    models = unique_models
                    
                    for model in models:
                        if not model in (None, '', 'undefined', 'none'):
                            vae_full_path = folder_paths.get_full_path("vae", model)
                            if not vae_full_path in (None, '', 'undefined', 'none'):
                                sha_result = get_sha256(vae_full_path)
                                vae_hash = sha_result[:10] if sha_result else None
                            if not vae_hash in (None, '', 'undefined', 'none'):
                                vae_file = return_filename_without_extension(model)
                                model_string[vae_file] = vae_hash

            # Build positive_for_meta and metadata extractor
            if not lora_names in (None, '', 'undefined', 'none'):
                lora_tokens, lora_weights = parse_lora_string(lora_names)
                # For metadata extraction, always use positive + lora_tokens to detect loras
                positive_with_loras = positive + str(lora_tokens)
                metadata_extractor = PromptMetadataExtractor([positive_with_loras, negative])
                # Only add loras to the actual prompt text if the option is enabled
                if add_loras_to_prompt:
                    positive_for_meta = positive_with_loras
                else:
                    positive_for_meta = positive
            else:
                positive_for_meta = positive
                lora_weights = {}
                metadata_extractor = PromptMetadataExtractor([positive, negative])

            embeddings = metadata_extractor.get_embeddings()
            loras = metadata_extractor.get_loras()

            if not sampler_name in (None, '', 'undefined', 'none'):
                if sampler_name is not None:
                    civitai_sampler_name = self.get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler)
                else:
                    civitai_sampler_name = "Euler Simple"
            else:
                civitai_sampler_name = "Euler Simple"

            extension_hashes = json.dumps(model_string | embeddings | loras)

            clip_skip_meta = global_values.get('clip_skip', '')
            try:
                if clip_skip_meta in (None, '', 'None'):
                    clip_skip_value_for_meta = None
                else:
                    clip_skip_value_for_meta = int(float(clip_skip_meta))
                    if clip_skip_value_for_meta < 0:
                        clip_skip_value_for_meta = abs(clip_skip_value_for_meta)
            except Exception:
                clip_skip_value_for_meta = None

            clip_skip_segment = f", Clip skip: {clip_skip_value_for_meta}" if clip_skip_value_for_meta is not None else ''
            
            # Normalize numeric/string-like values for safe formatting and metadata embedding
            def _val_to_str(v):
                try:
                    if v in (None, '', 'undefined', 'none'):
                        return ''
                except Exception:
                    pass
                try:
                    # Torch tensor with single value
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            return str(v.item())
                        else:
                            return str(v.tolist())
                    # numpy scalar
                    if isinstance(v, np.generic):
                        return str(np.asscalar(v)) if hasattr(np, 'asscalar') else str(v.item())
                except Exception:
                    pass
                return str(v)
            
            if not remove_prompts:
                positive_a111_params = handle_whitespace(positive_for_meta)
                negative_a111_params = f"\nNegative prompt: {handle_whitespace(negative)}"

                steps_str = _val_to_str(steps)
                cfg_str = _val_to_str(cfg)
                seed_str = _val_to_str(seed_value)

                a111_params = f"{positive_a111_params}{negative_a111_params}\nSteps: {steps_str}, Sampler: {civitai_sampler_name}, CFG scale: {cfg_str}, Seed: {seed_str}, Size: {width}x{height}{clip_skip_segment}, Hashes: {extension_hashes}, Version: ComfyUI"
            else:
                positive_a111_params = ''
                negative_a111_params = f"\nNegative prompt: "
                steps_str = _val_to_str(steps)
                cfg_str = _val_to_str(cfg)
                seed_str = _val_to_str(seed_value)
                a111_params = f"{positive_a111_params}{negative_a111_params}\nSteps: {steps_str}, Sampler: {civitai_sampler_name}, CFG scale: {cfg_str}, Seed: {seed_str}, Size: {width}x{height}, Clip skip: {clip_skip_meta}, Hashes: {extension_hashes}, Version: ComfyUI"

        delimiter = filename_delimiter

        # Store original images tensor for return value
        original_images_tensor = images

        # If no images were passed directly, require the pipe to supply images
        if images is None and pipe_opt is not None:
            # ctx should be set earlier when pipe_opt is processed
            try:
                pipe_images = ctx.get("images") if isinstance(ctx, dict) else None
            except Exception:
                pipe_images = None

            # Explicit emptiness checks to avoid ambiguous truthiness on tensors/arrays
            if pipe_images is None:
                raise RuntimeError("RvImage_SaveImages: pipe_opt provided but contains no 'images' data. Provide images via the 'images' input or include an 'images' key in the pipe.")

            # Store the original tensor for return value
            original_images_tensor = pipe_images

            # Normalize common container types into a list of per-image objects
            if isinstance(pipe_images, torch.Tensor):
                # Tensor may be (N, C, H, W) or (C, H, W)
                if pipe_images.numel() == 0:
                    raise RuntimeError("RvImage_SaveImages: pipe_opt provided but contains an empty tensor for 'images'.")
                if pipe_images.dim() == 4:
                    images = [pipe_images[i] for i in range(pipe_images.size(0))]
                else:
                    images = [pipe_images]
            elif isinstance(pipe_images, np.ndarray):
                if pipe_images.size == 0:
                    raise RuntimeError("RvImage_SaveImages: pipe_opt provided but contains an empty numpy array for 'images'.")
                if pipe_images.ndim == 4:
                    images = [pipe_images[i] for i in range(pipe_images.shape[0])]
                else:
                    images = [pipe_images]
            elif isinstance(pipe_images, (list, tuple)):
                if len(pipe_images) == 0:
                    raise RuntimeError("RvImage_SaveImages: pipe_opt provided but contains no 'images' data. Provide images via the 'images' input or include an 'images' key in the pipe.")
                images = list(pipe_images)
                # For lists/tuples, we need to stack them back into a tensor for return
                try:
                    original_images_tensor = torch.stack(images) if len(images) > 1 else images[0]
                except Exception:
                    # If stacking fails, keep as list but this may cause issues
                    original_images_tensor = images
            else:
                # Accept single PIL Image or other single-image objects and wrap into a list
                images = [pipe_images]
                original_images_tensor = pipe_images
        
        number_padding = filename_number_padding
        lossless_webp = (lossless_webp == True)
        optimize_image = (optimize_image == True)

        original_output = self.output_dir

        # Setup output path
        if output_path in [None, '', 'none', '.', './']:
            output_path = self.output_dir

        output_path = string_placeholder(output_path, True)

        # Always resolve to absolute path inside ComfyUI output folder
        comfy_output_dir = os.path.abspath(self.output_dir)
        if not os.path.isabs(output_path):
            output_path = os.path.normpath(output_path)
            if output_path.startswith('.' + os.sep):
                output_path = output_path[2:]
            output_path = os.path.join(comfy_output_dir, output_path)
        output_path = os.path.abspath(output_path)

        # Force output_path to be inside comfy_output_dir
        if not output_path.startswith(comfy_output_dir):
            # Remove drive letter if present and join with comfy_output_dir
            rel_path = os.path.relpath(output_path, start=os.path.splitdrive(output_path)[0] or '/')
            output_path = os.path.join(comfy_output_dir, rel_path)
            output_path = os.path.abspath(output_path)

        # Check output destination
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                cstr(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.').warning.print()
                os.makedirs(output_path, exist_ok=True)

        filename_prefix = string_placeholder(filename_prefix, False)
        
        # Find existing counter values
        if filename_number_start:
            pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"
        existing_counters = []
        for filename in os.listdir(output_path):
            match = re.search(pattern, filename)
            if match and re.match(pattern, os.path.basename(filename)):
                existing_counters.append(int(match.group(1)))
        existing_counters.sort(reverse=True)

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set Extension
        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            cstr(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}").error.print()
            file_extension = ".png"

        results = list()
        output_files = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate metadata/pnginfo
            if extension == 'webp':
                img_exif = img.getexif()
                if embed_workflow:
                    workflow_metadata = ''
                    prompt_str = ''
                    if prompt is not None:
                        prompt_str = json.dumps(prompt)
                        img_exif[0x010f] = "Prompt:" + prompt_str
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            workflow_metadata += json.dumps(extra_pnginfo[x])
                    img_exif[0x010e] = "Workflow:" + workflow_metadata
                # Debug: show parameters string for webp (webp branch does not currently embed parameters)
                try:
                    cstr(f"WEBP parameters (diagnostic): {a111_params}").debug.print()
                except Exception:
                    pass
                exif_data: Union[bytes, PngInfo] = img_exif.tobytes()
            else:
                metadata = PngInfo()

                if embed_workflow:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                if pipe_opt != None and save_generation_data:
                    metadata.add_text("parameters", a111_params)
                    # Add a machine-readable lora weights JSON key so other tools
                    # can easily read numeric strengths without parsing the prompt.
                    if lora_weights:
                        try:
                            metadata.add_text('lora_weights', json.dumps(lora_weights))
                        except Exception as e:
                            cstr(f"Failed to add lora_weights metadata: {e}").error.print()

                exif_data = metadata

            # Delegate the filename stuffs
            if filename_number_start == True:
                file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                jsonfile = f"{counter:0{number_padding}}{delimiter}{filename_prefix}"
            else:
                file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                jsonfile = f"{filename_prefix}{delimiter}{counter:0{number_padding}}"
            if os.path.exists(os.path.join(output_path, file)):
                counter += 1

            # Save the images
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file,
                             quality=quality, optimize=optimize_image, dpi=(dpi, dpi))
                elif extension == 'webp':
                    img.save(output_file,
                             quality=quality, lossless=lossless_webp, exif=exif_data)
                elif extension == 'png':
                    img.save(output_file,
                             pnginfo=exif_data, optimize=optimize_image)
                elif extension == 'bmp':
                    img.save(output_file)
                elif extension == 'tiff':
                    img.save(output_file,
                             quality=quality, optimize=optimize_image)
                else:
                    img.save(output_file,
                             pnginfo=exif_data, optimize=optimize_image)

                cstr(f"Image file saved to: {output_file}").msg.print()
                output_files.append(output_file)
                   
                if show_previews:
                    subfolder = self.get_subfolder_path(output_file, original_output)
                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": self.type
                    })

            except OSError as e:
                cstr(f'Unable to save file to: {output_file}').error.print()
                cstr(str(e)).error.print()
            except Exception as e:
                cstr('Unable to save file due to the to the following error:').error.print()
                cstr(str(e)).error.print()

            if save_workflow_as_json:
                output_json = os.path.abspath(os.path.join(output_path, jsonfile))
                save_json(extra_pnginfo, output_json)
                #output_files.append(jsonfile + ".json")

            counter += 1

        filtered_paths: list[str] = []

        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type
                }
                results.append(image_data)

        # Purge VRAM if enabled and requested
        if enabled and Purge_VRAM:
            purge_vram()

        if show_previews == True:
            return {"ui": {"images": results, "files": output_files}, "result": (original_images_tensor, output_files,)}
        else:
            return {"ui": {"images": []}, "result": (original_images_tensor, output_files,)}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

NODE_NAME = 'Save Images [Eclipse]'
NODE_DESC = 'Save Images'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvImage_SaveImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
