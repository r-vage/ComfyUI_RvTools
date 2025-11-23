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

import comfy
from types import ModuleType
from typing import Optional

class AnyType(str):
    # A special class that is always equal in not-equal comparisons. Credit to pythongosssss

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False
  
class cstr(str):
    class color:
        END = '\x1b[0m'
        BOLD = '\x1b[1m'
        ITALIC = '\x1b[3m'
        UNDERLINE = '\x1b[4m'
        BLINK = '\x1b[5m'
        BLINK2 = '\x1b[6m'
        SELECTED = '\x1b[7m'

        BLACK = '\x1b[30m'
        RED = '\x1b[31m'
        GREEN = '\x1b[32m'
        YELLOW = '\x1b[33m'
        BLUE = '\x1b[34m'
        VIOLET = '\x1b[35m'
        BEIGE = '\x1b[36m'
        WHITE = '\x1b[37m'

        BLACKBG = '\x1b[40m'
        REDBG = '\x1b[41m'
        GREENBG = '\x1b[42m'
        YELLOWBG = '\x1b[43m'
        BLUEBG = '\x1b[44m'
        VIOLETBG = '\x1b[45m'
        BEIGEBG = '\x1b[46m'
        WHITEBG = '\x1b[47m'

        GREY = '\x1b[90m'
        LIGHTRED = '\x1b[91m'
        LIGHTGREEN = '\x1b[92m'
        LIGHTYELLOW = '\x1b[93m'
        LIGHTBLUE = '\x1b[94m'
        LIGHTVIOLET = '\x1b[95m'
        LIGHTBEIGE = '\x1b[96m'
        LIGHTWHITE = '\x1b[97m'

        GREYBG = '\x1b[100m'
        LIGHTREDBG = '\x1b[101m'
        LIGHTGREENBG = '\x1b[102m'
        LIGHTYELLOWBG = '\x1b[103m'
        LIGHTBLUEBG = '\x1b[104m'
        LIGHTVIOLETBG = '\x1b[105m'
        LIGHTBEIGEBG = '\x1b[106m'
        LIGHTWHITEBG = '\x1b[107m'

        @staticmethod
        def add_code(name: str, code: str):
#             Add a custom color code at runtime.
#
#             Raises ValueError if the code name already exists.
#
            key = name.upper()
            if not hasattr(cstr.color, key):
                setattr(cstr.color, key, code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text: str, suffix: str = ""):
        combined = f"{text}: {suffix}" if suffix else text
        return super().__new__(cls, combined)

    def __getattr__(self, attr: str):
#         Support attribute-based colorization and class-level access.
#
#         Examples:
#           cstr('hello').RED  -> wraps with RED/END
#           cstr('foo __NAME__')._cstrNAME  -> replaces __NAME__ with the code
#
        # Handle literal placeholder prefix '_cstr' (exact prefix)
        try:
            if attr.startswith("_cstr"):
                code_name = attr[len("_cstr") :].upper()
                code = getattr(self.color, code_name, None)
                if code is None:
                    raise AttributeError(f"color code '{code_name}' not found")
                modified_text = self.replace(f"__{code_name}__", f"{code}")
                return cstr(modified_text)

            # Direct color attribute (e.g. .RED)
            code = getattr(self.color, attr.upper(), None)
            if code is not None:
                modified_text = f"{code}{self}{self.color.END}"
                return cstr(modified_text)

            # Expose class-level helpers (if any)
            if hasattr(cstr, attr):
                return getattr(cstr, attr)
        except AttributeError:
            # Mirror normal attribute error semantics
            pass
        raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)


def purge_vram() -> None:
    # Central helper to purge VRAM and unload models safely.
    #
    # Use this from nodes instead of duplicating the try/except import and
    # GC/CUDA/model unload sequence. Any exception is reported via the
    # project's cstr warning helper so callers don't need to duplicate
    # error handling.
    #
    # This function unloads all models and clears allocator caches to free
    # maximum VRAM. This will require models to be reloaded on next use.
    # Based on comfyui-multigpu's soft_empty_cache_multigpu approach.
    try:
        import gc
        torch: Optional[ModuleType]
        comfy_mod: Optional[ModuleType]
        try:
            import torch  # type: ignore
        except Exception:
            torch = None

        try:
            import comfy.model_management  # type: ignore
            comfy_mod = comfy
        except Exception:
            comfy_mod = None

        # Step 1: Python garbage collection
        gc.collect()
        
        # Step 2: Clear device caches (multi-device support)
        if torch is not None:
            try:
                # CUDA devices
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    for i in range(device_count):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, 'ipc_collect'):
                                torch.cuda.ipc_collect()
                
                # MPS (Apple Silicon)
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                # XPU (Intel)
                if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()
                
                # NPU (Huawei/Ascend)
                if hasattr(torch, 'npu') and hasattr(torch.npu, 'empty_cache'):
                    torch.npu.empty_cache()
                
                # MLU (Cambricon)
                if hasattr(torch, 'mlu') and hasattr(torch.mlu, 'empty_cache'):
                    torch.mlu.empty_cache()
                
            except Exception:
                # Ignore device-specific failures
                pass

        # Step 3: ComfyUI model unloading and cache clearing
        if comfy_mod is not None:
            try:
                # Unload all models first, then clear caches
                if hasattr(comfy_mod.model_management, 'unload_all_models'):
                    comfy_mod.model_management.unload_all_models()
                if hasattr(comfy_mod.model_management, 'soft_empty_cache'):
                    comfy_mod.model_management.soft_empty_cache()
            except Exception:
                # Ignore model-management failures
                pass
    except Exception as e:
        try:
            cstr(f"VRAM purge failed: {e}").warning.print()
        except Exception:
            try:
                print(f"VRAM purge failed: {e}")
            except Exception:
                pass

# Resolution presets and mappings for image generation
RESOLUTION_PRESETS = [
    "Custom",
    "512x512 (1:1)",
    "512x682 (3:4)",
    "512x768 (2:3)",
    "512x910 (9:16)",
    "512x952 (1:1.85)",
    "512x1024 (1:2)",
    "512x1224 (1:2.39)",
    "640x1536 (9:21)",
    "768x1280 (3:5 Flux)",
    "768x1344 (9:16 HiDream)",
    "832x1216 (2:3 Flux, SDXL)",
    "832x1408 (1:1.692 HiDream)",
    "896x1152 (3:4)",
    "896x1536 (7:12 HiDream)",
    "1024x1024 (1:1)",
    "1024x1536 (2:3 Flux, Qwen)",
    "1024x2048 (1:2 Qwen)",
    "1152x896 (4:3)",
    "682x512 (4:3)",
    "768x512 (3:2)",
    "910x512 (16:9)",
    "952x512 (1.85:1)",
    "1024x512 (2:1)",
    "1224x512 (2.39:1)",
    "1536x640 (21:9)",
    "1280x768 (5:3 Flux)",
    "1344x768 (16:9 HiDream)",
    "1216x832 (3:2 Flux, SDXL)",
    "1408x832 (1.692:1 HiDream)",
    "1536x896 (12:7 HiDream)",
    "1536x1024 (3:2 Flux, Qwen)",
    "2048x1024 (2:1 Qwen)",
]

RESOLUTION_MAP = {
    "512x512 (1:1)": (512, 512),
    "512x682 (3:4)": (512, 682),
    "512x768 (2:3)": (512, 768),
    "512x910 (9:16)": (512, 910),
    "512x952 (1:1.85)": (512, 952),
    "512x1024 (1:2)": (512, 1024),
    "512x1224 (1:2.39)": (512, 1224),
    "640x1536 (9:21)": (640, 1536),
    "768x1280 (3:5 Flux)": (768, 1280),
    "768x1344 (9:16 HiDream)": (768, 1344),
    "832x1216 (2:3 Flux, SDXL)": (832, 1216),
    "832x1408 (1:1.692 HiDream)": (832, 1408),
    "896x1152 (3:4)": (896, 1152),
    "896x1536 (7:12 HiDream)": (896, 1536),
    "1024x1024 (1:1)": (1024, 1024),
    "1024x1536 (2:3 Flux, Qwen)": (1024, 1536),
    "1024x2048 (1:2 Qwen)": (1024, 2048),
    "1152x896 (4:3)": (1152, 896),
    "682x512 (4:3)": (682, 512),
    "768x512 (3:2)": (768, 512),
    "910x512 (16:9)": (910, 512),
    "952x512 (1.85:1)": (952, 512),
    "1024x512 (2:1)": (1024, 512),
    "1224x512 (2.39:1)": (1224, 512),
    "1536x640 (21:9)": (1536, 640),
    "1280x768 (5:3 Flux)": (1280, 768),
    "1344x768 (16:9 HiDream)": (1344, 768),
    "1216x832 (3:2 Flux, SDXL)": (1216, 832),
    "1408x832 (1.692:1 HiDream)": (1408, 832),
    "1536x896 (12:7 HiDream)": (1536, 896),
    "1536x1024 (3:2 Flux, Qwen)": (1536, 1024),
    "2048x1024 (2:1 Qwen)": (2048, 1024),
}

# Sampler and scheduler lists for ComfyUI
SAMPLERS_COMFY = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS_ANY = comfy.samplers.KSampler.SCHEDULERS #+ ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma', 'simple_test']


def copy_prompt_files_once(source_dir: str, target_dir: str) -> bool:
    # Copy Smart Prompt files from source to target directory if target doesn't exist.
    # This is a one-time operation to enable wildcard integration.
    #
    # Args:
    #     source_dir: Source directory path (ComfyUI_Eclipse/templates/prompt/)
    #     target_dir: Target directory path (ComfyUI/models/wildcards/smartprompt/)
    #
    # Returns:
    #     True if copy was successful or target already exists, False on error
    import os
    import shutil
    
    # If target already exists, skip copy
    if os.path.exists(target_dir):
        return True
    
    # If source doesn't exist, nothing to copy
    if not os.path.exists(source_dir):
        cstr(f"Smart Prompt source directory not found: {source_dir}").warning.print()
        return False
    
    try:
        # Create target directory and copy all contents
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy directory tree
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, target_item)
        
        cstr(f"Smart Prompt files copied to wildcards folder for wildcard integration").msg.print()
        return True
        
    except Exception as e:
        cstr(f"Failed to copy Smart Prompt files to wildcards: {e}").warning.print()
        return False


def create_junction(source_dir: str, link_dir: str) -> bool:
    # Create a junction (Windows) or symlink (Linux/macOS) from link_dir to source_dir.
    # This enables wildcards integration without file duplication.
    #
    # Args:
    #     source_dir: Target directory path (models/Eclipse/smart_prompt/)
    #     link_dir: Junction/symlink path (models/wildcards/smart_prompt/)
    #
    # Returns:
    #     True if junction created successfully or already exists, False on error
    import os
    import platform
    import subprocess
    
    # If link already exists, skip creation
    if os.path.exists(link_dir):
        return True
    
    # If source doesn't exist, can't create junction
    if not os.path.exists(source_dir):
        cstr(f"Junction source directory not found: {source_dir}").warning.print()
        return False
    
    try:
        # Create parent directory if needed
        parent_dir = os.path.dirname(link_dir)
        os.makedirs(parent_dir, exist_ok=True)
        
        system = platform.system()
        
        if system == "Windows":
            # Use mklink /J for directory junction on Windows
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", link_dir, source_dir],
                check=True,
                capture_output=True
            )
            cstr(f"Created junction: wildcards/smart_prompt → Eclipse/smart_prompt").msg.print()
        else:
            # Use ln -s for symbolic link on Linux/macOS
            os.symlink(source_dir, link_dir, target_is_directory=True)
            cstr(f"Created symlink: wildcards/smart_prompt → Eclipse/smart_prompt").msg.print()
        
        return True
        
    except Exception as e:
        # Silent failure - junction is optional for wildcards integration
        cstr(f"Could not create junction for wildcards integration (optional): {e}").warning.print()
        return False


def migrate_old_folders(comfyui_root: str) -> None:
    # Migrate user files from old folder structure to new Eclipse structure.
    # This is a one-time migration to preserve user customizations.
    #
    # Old locations:
    #   - models/smart_loader_templates → models/Eclipse/loader_templates
    #   - models/wildcards/smartprompt → models/Eclipse/smart_prompt
    #
    # Args:
    #     comfyui_root: ComfyUI root directory path
    import os
    import shutil
    
    migrations = [
        {
            'old': os.path.join(comfyui_root, 'models', 'smart_loader_templates'),
            'new': os.path.join(comfyui_root, 'models', 'Eclipse', 'loader_templates'),
            'name': 'Smart Loader templates'
        },
        {
            'old': os.path.join(comfyui_root, 'models', 'wildcards', 'smartprompt'),
            'new': os.path.join(comfyui_root, 'models', 'Eclipse', 'smart_prompt'),
            'name': 'Smart Prompt files'
        }
    ]
    
    for migration in migrations:
        old_path = migration['old']
        new_path = migration['new']
        name = migration['name']
        
        # Skip if old location doesn't exist
        if not os.path.exists(old_path):
            continue
        
        # Skip if new location already has content (already migrated or fresh install)
        if os.path.exists(new_path) and os.listdir(new_path):
            try:
                # Clean up old location if new location exists
                shutil.rmtree(old_path)
                cstr(f"Removed old {name} folder (migrated previously)").msg.print()
            except Exception as e:
                cstr(f"Could not remove old {name} folder: {e}").warning.print()
            continue
        
        try:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Move the entire directory to new location
            if os.path.exists(new_path):
                # New path exists but is empty, remove it first
                shutil.rmtree(new_path)
            
            shutil.move(old_path, new_path)
            cstr(f"Migrated {name} to Eclipse folder").msg.print()
            
        except Exception as e:
            # If move fails, try copy and delete
            try:
                os.makedirs(new_path, exist_ok=True)
                
                # Copy directory tree
                for item in os.listdir(old_path):
                    source_item = os.path.join(old_path, item)
                    target_item = os.path.join(new_path, item)
                    
                    if os.path.isdir(source_item):
                        shutil.copytree(source_item, target_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_item, target_item)
                
                # Remove old directory after successful copy
                shutil.rmtree(old_path)
                cstr(f"Migrated {name} to Eclipse folder (via copy)").msg.print()
                
            except Exception as copy_error:
                cstr(f"Failed to migrate {name}: {copy_error}").warning.print()
