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
from ..core import CATEGORY

class RvFolder_AddFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}, "Base path to which the folder will be added.\nType-safe: handles None and empty input."),
                "folder_name": ("STRING", {"multiline": False, "default": "SubFolder"}, "Folder name to join to the base path.\nType-safe: handles None and empty input."),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.FOLDER.value
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)    
    
    FUNCTION = "execute"

    def execute(self, path, folder_name):
        # Joins a folder name to a base path, returning the new path as a string.
        # Handles None and empty input robustly.

        if not isinstance(path, str) or not path:
            return ("",)
        if not isinstance(folder_name, str) or not folder_name:
            return (path,)
        new_path = os.path.join(path, folder_name)
        return (new_path,)

NODE_NAME = 'Add Folder [RvTools]'
NODE_DESC = 'Add Folder'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvFolder_AddFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}