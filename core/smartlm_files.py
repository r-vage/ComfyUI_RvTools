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

# Smart Language Model File Handling
# Handles file scanning, model list generation, and download utilities

from pathlib import Path
from typing import List
from collections import defaultdict

from . import cstr


def download_with_progress(url: str, path: str, name: str) -> None:
    # Download file with progress bar
    import urllib.request
    from tqdm import tqdm
    
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc=f'[SmartLM] Downloading {name}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))


def get_llm_model_list() -> List[str]:
    # Scan models/LLM folder and return list of available models.
    # First collects all model files, then filters to show:
    # - For shard files: show folder/ instead of individual files
    # - For single files: show full relative path to the file
    try:
        import folder_paths
        llm_dir = Path(folder_paths.models_dir) / "LLM"
        
        if not llm_dir.exists():
            return ["(No models/LLM folder found)"]
        
        model_extensions = {'.safetensors', '.gguf', '.bin', '.pt'}
        all_model_files = []
        
        # Step 1: Recursively scan and collect all model files
        def scan_files(base_path: Path, relative_path: str = ""):
            """Recursively collect all model files"""
            try:
                for item in base_path.iterdir():
                    if item.is_file() and item.suffix in model_extensions:
                        # Build full relative path
                        if relative_path:
                            file_path = f"{relative_path}/{item.name}"
                        else:
                            file_path = item.name
                        all_model_files.append(file_path)
                    elif item.is_dir():
                        # Recurse into subdirectories (limit depth to avoid infinite loops)
                        item_rel_path = f"{relative_path}/{item.name}" if relative_path else item.name
                        if relative_path.count('/') < 4:  # Max 4 levels deep
                            scan_files(item, item_rel_path)
            except PermissionError:
                pass  # Skip directories we can't access
        
        scan_files(llm_dir)
        
        if not all_model_files:
            return ["(No models found in models/LLM)"]
        
        # Step 2: Group files by their parent folder
        folder_files = defaultdict(list)
        
        for file_path in all_model_files:
            if '/' in file_path:
                folder = file_path.rsplit('/', 1)[0]
                filename = file_path.rsplit('/', 1)[1]
            else:
                folder = ""  # Root level
                filename = file_path
            
            folder_files[folder].append(filename)
        
        # Step 3: Check for config.json to identify model repositories
        import folder_paths
        llm_base = Path(folder_paths.models_dir) / "LLM"
        
        folders_with_config = set()
        for folder in folder_files.keys():
            if folder:  # Skip root level
                config_path = llm_base / folder / "config.json"
                if config_path.exists():
                    folders_with_config.add(folder)
        
        # Step 4: Filter to create final model list
        models = []
        
        for folder, files in folder_files.items():
            # Separate mmproj files from model files
            model_files = [f for f in files if 'mmproj' not in f.lower()]
            
            # Check if any file is a shard file
            has_shards = any('-of-' in f or '.shard' in f.lower() for f in model_files)
            
            # Check if folder has config.json (indicates it's a model repository)
            has_config = folder in folders_with_config
            
            if has_shards or has_config:
                # Show folder/ for sharded models or model repositories with config.json
                if folder:
                    models.append(folder + "/")
                else:
                    # Shards in root - shouldn't happen but handle it
                    for f in model_files:
                        models.append(f)
            else:
                # List all non-shard model files individually (excluding mmproj)
                # These are typically single-file models like GGUF files
                for f in model_files:
                    if folder:
                        models.append(f"{folder}/{f}")
                    else:
                        models.append(f)
        
        return sorted(models)
    
    except Exception as e:
        cstr(f"[SmartLM] Error scanning models/LLM: {e}").error.print()
        return ["(Error scanning models folder)"]


def get_mmproj_list() -> List[str]:
    # Scan models/LLM folder for mmproj files for GGUF QwenVL models.
    # Returns only individual .mmproj files and .gguf files containing 'mmproj' in the name.
    # Never shows folders, only file paths.
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
                        if relative_path.count('/') < 4:  # Max 4 levels deep
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


def search_model_file(filename: str, llm_base: Path) -> Path | None:
    # Search recursively for a model file in the LLM folder.
    # Used to find legacy model files when template paths are outdated.
    # Returns Path object if found, None otherwise.
    try:
        if not llm_base.exists():
            return None
        
        # Search recursively (limit depth implicitly by rglob)
        for path in llm_base.rglob(filename):
            if path.is_file():
                return path
        
        return None
    except Exception as e:
        cstr(f"[SmartLM] Error searching for {filename}: {e}").warning.print()
        return None


def calculate_model_size(target_path: Path) -> float:
    # Calculate total model size in GB from a file or directory.
    # Handles sharded models, single files, and directories with multiple model files.
    # Returns size in GB, or 0.0 if calculation fails.
    try:
        total_size_gb = 0.0
        
        if target_path.is_file():
            # Single file (GGUF, safetensors, etc.)
            total_size_gb = target_path.stat().st_size / (1024**3)
        elif target_path.is_dir():
            # Model folder - check for sharded models first, then single files
            # Priority: .safetensors (preferred) > .bin > .pt > .gguf
            all_files = list(target_path.rglob('*'))
            model_files = [f for f in all_files if f.is_file()]
            
            # Check for shard files (e.g., model-00001-of-00005.safetensors)
            safetensors_files = [f for f in model_files if f.suffix == '.safetensors']
            bin_files = [f for f in model_files if f.suffix == '.bin']
            pt_files = [f for f in model_files if f.suffix == '.pt']
            gguf_files = [f for f in model_files if f.suffix == '.gguf']
            
            # Check if we have shards (files with -of- pattern)
            has_shards = lambda files: any('-of-' in f.name for f in files)
            
            # Priority: safetensors shards > single safetensors > bin shards > single bin > pt > gguf
            if has_shards(safetensors_files):
                # Use safetensors shards
                for file in safetensors_files:
                    if '-of-' in file.name:
                        total_size_gb += file.stat().st_size / (1024**3)
            elif safetensors_files:
                # Single safetensors file (no shards)
                for file in safetensors_files:
                    total_size_gb += file.stat().st_size / (1024**3)
            elif has_shards(bin_files):
                # Use bin shards
                for file in bin_files:
                    if '-of-' in file.name:
                        total_size_gb += file.stat().st_size / (1024**3)
            elif bin_files:
                # Single bin file
                for file in bin_files:
                    total_size_gb += file.stat().st_size / (1024**3)
            elif pt_files:
                # PT files
                for file in pt_files:
                    total_size_gb += file.stat().st_size / (1024**3)
            elif gguf_files:
                # GGUF files
                for file in gguf_files:
                    total_size_gb += file.stat().st_size / (1024**3)
        
        return total_size_gb
    
    except Exception as e:
        cstr(f"[SmartLM] Error calculating model size: {e}").warning.print()
        return 0.0
