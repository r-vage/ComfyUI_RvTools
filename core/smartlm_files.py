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
# Handles file scanning, model list generation, download utilities, and hash verification

from pathlib import Path
from typing import List
from collections import defaultdict
import hashlib

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


def calculate_file_hash(file_path: Path, show_progress: bool = True) -> str:
    """
    Calculate SHA256 hash of a file with optional progress display.
    
    Args:
        file_path: Path to the file to hash
        show_progress: Whether to display progress for large files (>100MB)
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    import sys
    
    sha256_hash = hashlib.sha256()
    file_size = file_path.stat().st_size
    bytes_processed = 0
    last_progress = -1
    
    # Show initial message with file size for large files
    size_mb = file_size / (1024 * 1024)
    if show_progress and file_size > 100 * 1024 * 1024:
        cstr(f"[SmartLM] Calculating hash for {file_path.name} ({size_mb:.1f} MB)...").msg.print()
    elif show_progress:
        cstr(f"[SmartLM] Calculating hash for {file_path.name}...").msg.print()
    
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):  # 8MB chunks for speed
            sha256_hash.update(chunk)
            bytes_processed += len(chunk)
            # Show progress for large files (> 100MB)
            if show_progress and file_size > 100 * 1024 * 1024:
                progress = int((bytes_processed / file_size) * 100)
                # Update every 1% to keep progress smooth
                if progress != last_progress:
                    # Use carriage return to overwrite the same line
                    sys.stdout.write(f"\rEclipse: [SmartLM]   Hashing: {progress}% ({bytes_processed / (1024*1024):.0f}/{size_mb:.0f} MB)")
                    sys.stdout.flush()
                    last_progress = progress
    
    # Print newline after progress is complete to preserve the final line
    if show_progress and file_size > 100 * 1024 * 1024:
        print()  # Move to next line
    
    return sha256_hash.hexdigest()


def verify_model_integrity(model_path: Path, repo_id: str = None, hf_filename: str = None) -> bool:
    """
    Verify model file integrity using SHA256 checksums.
    Calculates and saves hashes on first load, then verifies on subsequent loads.
    
    Args:
        model_path: Path to model file or directory
        repo_id: HuggingFace repo_id (user/repo format or full URL)
        hf_filename: Optional filename to use for HuggingFace lookup (for renamed files)
        
    Returns:
        True if verification passes or hash is newly calculated
        False if corruption detected
    """
    try:
        # Look for model.safetensors or pytorch_model.bin
        critical_files = []
        if model_path.is_dir():
            safetensors = list(model_path.glob("*.safetensors"))
            bin_files = list(model_path.glob("pytorch_model*.bin"))
            critical_files = safetensors if safetensors else bin_files
        else:
            critical_files = [model_path] if model_path.suffix in ['.gguf', '.safetensors', '.bin'] else []
        
        if not critical_files:
            cstr(f"[SmartLM] No model files found to verify at {model_path}").warning.print()
            return True  # Skip verification
        
        verified_count = 0
        failed_count = 0
        calculated_count = 0
        
        for file_path in critical_files:
            sha_file = file_path.parent / f"{file_path.name}.sha256"
            expected_hash = None
            
            # Check if we have a cached hash file first
            if sha_file.exists():
                try:
                    expected_hash = sha_file.read_text().strip().split()[0]
                    verified_count += 1
                    continue  # Skip hash calculation, already verified
                except:
                    pass
            
            # If no cached hash, try to get it from HuggingFace
            if not expected_hash and repo_id:
                try:
                    from huggingface_hub import hf_hub_url, get_hf_file_metadata
                    
                    # Use provided hf_filename if available (for renamed files), otherwise use local filename
                    lookup_filename = hf_filename if hf_filename else file_path.name
                    cstr(f"[SmartLM] Fetching hash from HuggingFace for {lookup_filename}...").msg.print()
                    
                    # Construct URL and get metadata
                    url = hf_hub_url(repo_id=repo_id, filename=lookup_filename, repo_type="model")
                    metadata = get_hf_file_metadata(url=url)
                    
                    # ETag is the SHA256 hash for git-lfs files (per HuggingFace docs)
                    if hasattr(metadata, 'etag') and metadata.etag:
                        expected_hash = metadata.etag
                        cstr(f"[SmartLM] Retrieved hash from HuggingFace").msg.print()
                    else:
                        cstr(f"[SmartLM] No hash available in HuggingFace metadata for {lookup_filename}").warning.print()
                except Exception as e:
                    cstr(f"[SmartLM] Could not retrieve hash from HuggingFace ({repo_id}/{lookup_filename}): {e}").warning.print()
            
            # If we still don't have a reference hash, skip verification
            if not expected_hash:
                cstr(f"[SmartLM] No reference hash available for {file_path.name}, skipping verification").warning.print()
                calculated_count += 1
                continue
            
            # Calculate actual hash using centralized function
            actual_hash = calculate_file_hash(file_path, show_progress=True)
            
            # Verify against HuggingFace hash
            if actual_hash == expected_hash:
                cstr(f"[SmartLM] ✓ {file_path.name} integrity verified").msg.print()
                verified_count += 1
                
                # Save hash file for future fast verification
                try:
                    sha_file.write_text(expected_hash)
                    cstr(f"[SmartLM] Cached hash to {sha_file.name}").msg.print()
                except Exception as e:
                    cstr(f"[SmartLM] Warning: Could not cache hash: {e}").warning.print()
            else:
                cstr(f"[SmartLM] ✗ {file_path.name} CORRUPTED! Hash mismatch.").error.print()
                cstr(f"[SmartLM]   Expected: {expected_hash}").error.print()
                cstr(f"[SmartLM]   Got:      {actual_hash}").error.print()
                cstr(f"[SmartLM]   Please delete the model folder and redownload.").error.print()
                failed_count += 1
                # Don't save hash file on failure - user needs to redownload
        
        if failed_count > 0:
            cstr(f"[SmartLM] ⚠ Model verification FAILED! {failed_count} corrupted file(s) detected.").error.print()
            cstr(f"[SmartLM] Please delete the model folder and redownload: {model_path}").error.print()
            return False
        elif verified_count > 0:
            cstr(f"[SmartLM] ✓ Model integrity verified ({verified_count} file(s))").msg.print()
        elif calculated_count > 0:
            cstr(f"[SmartLM] ⚠ No reference hash available, skipping verification for {calculated_count} file(s)").warning.print()
        
        return True
        
    except Exception as e:
        cstr(f"[SmartLM] Model verification error (non-critical): {e}").warning.print()
        return True  # Don't block loading on verification errors


def extract_repo_id_from_url(repo_id: str) -> str:
    """
    Extract actual repo_id (namespace/repo_name) from a HuggingFace URL.
    
    Args:
        repo_id: Either a direct repo_id like "user/repo" or a full HuggingFace URL
    
    Returns:
        Extracted repo_id in format "user/repo", or original string if not a URL
    
    Examples:
        "bartowski/model" -> "bartowski/model"
        "https://huggingface.co/user/repo/resolve/main/file.gguf" -> "user/repo"
    """
    if not repo_id:
        return ""
    
    # If it's a URL, extract the repo_id part
    if repo_id.startswith("http") and "huggingface.co" in repo_id:
        parts = repo_id.split('/')
        if len(parts) >= 5:
            # URL format: https://huggingface.co/USER/REPO/resolve/main/file
            return f"{parts[3]}/{parts[4]}"
    
    # Already in correct format or not a URL
    return repo_id
