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

from typing import Optional, Any
from ..core import CATEGORY, cstr
from ..core import AnyType
import re, os

any = AnyType("*")

class RvPipe_Out_VCNameGen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("pipe", {"tooltip": "Input pipe containing path, frame load cap, mask frames, and files."}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    # Support union of FilenameGenerator v1 and v2 outputs
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "BOOLEAN", "STRING", "STRING",)
    RETURN_NAMES = ("path", "rel_path", "frame_load_cap", "mask_first_frames", "mask_last_frames", "simple_combine", "files", "files_join",)
    FUNCTION = "execute"
    
    def execute(self, pipe: Optional[dict[Any, Any]] = None) -> tuple:
        # Expect a dict-style pipe with canonical keys. Tuples are no longer supported.
        if pipe is None:
            raise ValueError("Input pipe must not be None and must be a dict-style pipe")
        if not isinstance(pipe, dict):
            raise ValueError("RvPipe_Out_VCNameGen expects dict-style pipes only.")

        path = pipe.get("path", "")
        frame_load_cap = pipe.get("frame_load_cap", 0)
        mask_first_frames = pipe.get("mask_first_frames", 0)
        mask_last_frames = pipe.get("mask_last_frames", 0)
        file_dict = pipe.get("file_dict") or pipe.get("files")
        join_dict = pipe.get("join_dict") or pipe.get("files_join")

        files = ""
        files_join = ""
        simple_combine = pipe.get("simple_combine", False)
        rel_path = re.sub(r"(<?^.*output)", ".", path)

        try:
            cstr(f"rel_path: {rel_path}").msg.print()
        except Exception:
            pass

        if file_dict not in (None, '', 'undefined', 'none'):
            try:
                if file_dict is not None:
                    files = str(file_dict.get("FILE"))
            except Exception:
                files = str(file_dict)
            files = re.sub(r"^\[", "", files)
            files = re.sub(r"\]", "", files)
            files = re.sub(r"'", "", files)

        if join_dict not in (None, '', 'undefined', 'none'):
            try:
                if join_dict is not None:
                    files_join = str(join_dict.get("JOIN"))
            except Exception:
                files_join = str(join_dict)
            files_join = re.sub(r"^\[", "", files_join)
            files_join = re.sub(r"\]", "", files_join)
            files_join = re.sub(r"'", "", files_join)

        return (path, rel_path, frame_load_cap, mask_first_frames, mask_last_frames, simple_combine, files, files_join)

NODE_NAME = 'Pipe Out VC Name Generator [RvTools]'
NODE_DESC = 'Pipe Out VC Name Generator'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvPipe_Out_VCNameGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}