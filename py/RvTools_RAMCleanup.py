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

import psutil
import ctypes
import time
import platform
import subprocess
from ..core import CATEGORY
from ..core import AnyType
from ..core.common import cstr

# Import Windows-specific modules only on Windows
if platform.system() == "Windows":
    from ctypes import wintypes

any = AnyType("*")

class Eclipse_RAMCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "clean_file_cache": ("BOOLEAN", {"default": True, "label": "Clear File Cache"}),
                "clean_processes": ("BOOLEAN", {"default": True, "label": "Clear Process Memory"}),
                "clean_dlls": ("BOOLEAN", {"default": True, "label": "Clear Unused DLLs"}),
                "retry_times": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Retry Times"
                }),
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
    FUNCTION = "clean_ram"
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def get_ram_usage(self):
        # Get current RAM usage statistics
        memory = psutil.virtual_memory()
        return memory.percent, memory.available / (1024 * 1024)

    def get_detailed_memory_info(self):
        # Get detailed memory information
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 * 1024),  # MB
            'available': memory.available / (1024 * 1024),  # MB
            'used': memory.used / (1024 * 1024),  # MB
            'percent': memory.percent,
            'free': memory.free / (1024 * 1024)  # MB
        }

    def _clear_file_cache_windows(self):
        # Clear Windows file cache
        try:
            result = ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
            return result == 0
        except Exception as e:
            return False

    def _check_sudo_available(self):
        # Check if sudo is available and configured
        try:
            # Check if sudo command exists
            result = subprocess.run(
                ["which", "sudo"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                return False, "sudo command not found"
            
            # Check if sudo is configured (will return 0 if passwordless sudo is set up)
            result = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                return False, "sudo requires password (configure passwordless sudo or run ComfyUI with sudo)"
            
            return True, "sudo available"
        except (subprocess.TimeoutExpired, Exception) as e:
            return False, f"sudo check failed: {str(e)}"

    def _clear_file_cache_linux(self):
        # Clear Linux file cache using multiple methods
        # Check if sudo is available first
        sudo_ok, sudo_msg = self._check_sudo_available()
        if not sudo_ok:
            cstr(f"Cannot clear file cache on Linux: {sudo_msg}").warning.print()
            return False
        
        methods = [
            (["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], "sudo echo method"),
            (["sudo", "-n", "sysctl", "vm.drop_caches=3"], "sysctl method"),
        ]

        for cmd, method_name in methods:
            try:
                result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, Exception):
                continue

        return False

    def _clear_process_memory_windows(self):
        # Clear working set of user processes (safely)
        cleaned_count = 0

        # System processes to avoid (case-insensitive)
        system_processes = {
            'system', 'system idle process', 'svchost.exe', 'csrss.exe', 'wininit.exe',
            'winlogon.exe', 'lsass.exe', 'services.exe', 'smss.exe', 'explorer.exe'
        }

        try:
            for process in psutil.process_iter(['pid', 'name']):
                try:
                    process_name = process.info['name']
                    if process_name and process_name.lower() in system_processes:
                        continue

                    # Only clean processes that are not system critical
                    handle = ctypes.windll.kernel32.OpenProcess(
                        wintypes.DWORD(0x001F0FFF),  # PROCESS_ALL_ACCESS
                        wintypes.BOOL(False),
                        wintypes.DWORD(process.info['pid'])
                    )

                    if handle:
                        result = ctypes.windll.psapi.EmptyWorkingSet(handle)
                        ctypes.windll.kernel32.CloseHandle(handle)
                        if result == 0:  # Success
                            cleaned_count += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    # Skip processes we can't access or that disappear
                    continue
                except Exception:
                    continue

            return cleaned_count

        except Exception:
            return 0

    def _clear_dlls_windows(self):
        # Clear current process working set
        try:
            # This affects the current Python process
            result = ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            return result == 0
        except Exception:
            return False

    def _clear_dlls_linux(self):
        # Sync filesystem buffers on Linux
        try:
            subprocess.run(["sync"], check=True, capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, Exception):
            return False

    def clean_ram(self, anything, clean_file_cache, clean_processes, clean_dlls, retry_times, unique_id=None, extra_pnginfo=None):
        # Main RAM cleanup function with improved error handling and safety
        if retry_times < 1 or retry_times > 10:
            cstr(f"Invalid retry_times value: {retry_times}. Using default of 3.").warning.print()
            retry_times = 3

        try:
            initial_mem = self.get_detailed_memory_info()
            system = platform.system()
            
            # Start message
            cstr(f"=== RAM Cleanup Started ===").msg.print()
            cstr(f"Initial - Usage: {initial_mem['percent']:.1f}% | Available: {initial_mem['available']:.1f}MB | Total: {initial_mem['total']:.1f}MB").msg.print()
            
            total_cleaned_processes = 0
            operations_completed = set()
            attempt_details = []

            for attempt in range(retry_times):
                attempt_operations = []
                attempt_cleaned_processes = 0

                # File cache cleanup
                if clean_file_cache:
                    if system == "Windows":
                        if self._clear_file_cache_windows():
                            attempt_operations.append("Cache")
                            operations_completed.add("File Cache")
                    elif system == "Linux":
                        if self._clear_file_cache_linux():
                            attempt_operations.append("Cache")
                            operations_completed.add("File Cache")

                # Process memory cleanup
                if clean_processes:
                    if system == "Windows":
                        cleaned = self._clear_process_memory_windows()
                        if cleaned > 0:
                            attempt_cleaned_processes = cleaned
                            total_cleaned_processes += cleaned
                            attempt_operations.append(f"Proc({cleaned})")
                            operations_completed.add("Processes")

                # DLL/working set cleanup
                if clean_dlls:
                    if system == "Windows":
                        if self._clear_dlls_windows():
                            attempt_operations.append("DLL")
                            operations_completed.add("DLL Working Set")
                    elif system == "Linux":
                        if self._clear_dlls_linux():
                            attempt_operations.append("Sync")
                            operations_completed.add("Filesystem Sync")

                # Store attempt details
                current_mem = self.get_detailed_memory_info()
                attempt_details.append({
                    'attempt': attempt + 1,
                    'operations': attempt_operations,
                    'usage': current_mem['percent'],
                    'available': current_mem['available']
                })

                # Brief pause between attempts
                if attempt < retry_times - 1:
                    time.sleep(0.5)

            # Final summary with all attempts
            final_mem = self.get_detailed_memory_info()
            memory_change = final_mem['available'] - initial_mem['available']
            
            # Build single consolidated message
            summary_lines = ["\n=== RAM Cleanup Progress ==="]
            
            for detail in attempt_details:
                ops = ', '.join(detail['operations']) if detail['operations'] else 'None'
                summary_lines.append(
                    f"Attempt {detail['attempt']}/{retry_times}: [{ops}] → "
                    f"Usage: {detail['usage']:.1f}% | Available: {detail['available']:.1f}MB"
                )
            
            summary_lines.append("\n=== Cleanup Complete ===")
            summary_lines.append(f"Operations: {', '.join(sorted(operations_completed)) if operations_completed else 'None'}")
            summary_lines.append(f"Processes Cleaned: {total_cleaned_processes}")
            summary_lines.append(f"Memory Change: {memory_change:+.1f}MB")
            summary_lines.append(f"Final - Usage: {final_mem['percent']:.1f}% | Available: {final_mem['available']:.1f}MB")
            
            # Print single consolidated message
            cstr("\n".join(summary_lines)).msg.print()

        except Exception as e:
            cstr(f"Critical error during RAM cleanup process: {e}").error.print()
            import traceback
            traceback.print_exc()

        return (anything,)
    

NODE_NAME = 'RAM Cleanup [Eclipse]'
NODE_DESC = 'RAM Cleanup'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: Eclipse_RAMCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}