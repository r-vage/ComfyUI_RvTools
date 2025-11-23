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
#
# Credits to LAOGOU-666: https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git
# improved and adapted for Comfyui_Eclipse

import time
import gc
import torch
from server import PromptServer
from ..core import CATEGORY
from ..core import AnyType
from ..core.common import cstr

any = AnyType("*")

class Eclipse_VRAMCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "offload_model": ("BOOLEAN", {"default": True, "label": "Offload Models", "tooltip": "Unload models from VRAM via ComfyUI"}),
                "offload_cache": ("BOOLEAN", {"default": True, "label": "Clear VRAM Cache", "tooltip": "Clear VRAM cache via ComfyUI"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": False, "label": "Aggressive Cleanup", "tooltip": "Force PyTorch CUDA cache clear and garbage collection (may cause brief lag)"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def _validate_prompt_server(self):
        # Validate that PromptServer is available and accessible
        try:
            if not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
                return False, "PromptServer instance not available"
            return True, None
        except Exception as e:
            return False, f"PromptServer validation error: {e}"

    def _send_cleanup_signal(self, offload_model, offload_cache):
        # Send cleanup signal to ComfyUI frontend
        signal_data = {
            "type": "cleanup_request",
            "data": {
                "unload_models": bool(offload_model),
                "free_memory": bool(offload_cache)
            }
        }

        try:
            PromptServer.instance.send_sync("memory_cleanup", signal_data)
            return True, signal_data
        except AttributeError as e:
            return False, f"PromptServer method not available: {e}"
        except Exception as e:
            return False, f"Failed to send cleanup signal: {e}"

    def _aggressive_vram_cleanup(self):
        # Perform aggressive VRAM cleanup using PyTorch and garbage collection
        try:
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            
            # Get initial VRAM usage
            initial_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            initial_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            # Get final VRAM usage
            final_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            final_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            freed_allocated = initial_allocated - final_allocated
            freed_reserved = initial_reserved - final_reserved
            
            return True, f"Freed {freed_allocated:.1f}MB allocated, {freed_reserved:.1f}MB reserved"
        except Exception as e:
            return False, f"Aggressive cleanup failed: {str(e)}"

    def empty_cache(self, anything, offload_model, offload_cache, aggressive_cleanup, unique_id=None, extra_pnginfo=None):
        # Send VRAM cleanup signal to ComfyUI frontend with validation and feedback
        start_time = time.time()

        try:
            # Validate inputs
            if not isinstance(offload_model, bool):
                offload_model = bool(offload_model)
            if not isinstance(offload_cache, bool):
                offload_cache = bool(offload_cache)
            if not isinstance(aggressive_cleanup, bool):
                aggressive_cleanup = bool(aggressive_cleanup)

            # Build status message
            operations = []
            if offload_model:
                operations.append("Offload Models")
            if offload_cache:
                operations.append("Clear Cache")
            if aggressive_cleanup:
                operations.append("Aggressive Cleanup")
            
            if not operations:
                cstr("=== VRAM Cleanup Skipped ===").msg.print()
                cstr("No cleanup operations selected").msg.print()
                return (anything,)

            # Start message
            cstr("=== VRAM Cleanup Started ===").msg.print()
            cstr(f"Operations: {', '.join(operations)}").msg.print()

            status_messages = []
            
            # Standard cleanup via PromptServer
            if offload_model or offload_cache:
                # Validate PromptServer availability
                server_ok, server_error = self._validate_prompt_server()
                if not server_ok:
                    status_messages.append(f"ComfyUI Signal: Failed - {server_error}")
                else:
                    # Send cleanup signal
                    signal_sent, signal_result = self._send_cleanup_signal(offload_model, offload_cache)
                    if signal_sent:
                        time.sleep(0.5)  # Brief pause for frontend processing
                        status_messages.append("ComfyUI Signal: Success")
                    else:
                        status_messages.append(f"ComfyUI Signal: Failed - {signal_result}")
            
            # Aggressive cleanup
            if aggressive_cleanup:
                aggressive_ok, aggressive_msg = self._aggressive_vram_cleanup()
                if aggressive_ok:
                    status_messages.append(f"Aggressive: {aggressive_msg}")
                else:
                    status_messages.append(f"Aggressive: {aggressive_msg}")
            
            elapsed = time.time() - start_time
            
            # Consolidated output
            cstr(f"Status: {', '.join(status_messages)}").msg.print()
            cstr(f"Time: {elapsed:.2f}s").msg.print()
            cstr("=== VRAM Cleanup Complete ===").msg.print()

        except Exception as e:
            elapsed = time.time() - start_time
            cstr(f"Status: Error - {str(e)}").error.print()
            cstr(f"Time: {elapsed:.2f}s").msg.print()
            cstr("=== VRAM Cleanup Complete ===").msg.print()

        return (anything,)


NODE_NAME = 'VRAM Cleanup [Eclipse]'
NODE_DESC = 'VRAM Cleanup'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: Eclipse_VRAMCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}