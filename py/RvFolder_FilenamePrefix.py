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
from datetime import datetime
from ..core import CATEGORY, AnyType

any = AnyType("*")

def format_datetime(datetime_format):
    today = datetime.now()
    try:
        timestamp = today.strftime(datetime_format)
    except:
        timestamp = today.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def format_date_time(string, position, datetime_format):
    today = datetime.now()
    if position == "prefix":
        return f"{today.strftime(datetime_format)}_{string}"
    if position == "postfix":
        return f"{string}_{today.strftime(datetime_format)}"

class RvFolder_FilenamePrefix:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name_prefix": ("STRING", {"multiline": False, "default": "image"}, "Filename prefix to join to the base path. Type-safe: handles None and empty input."),
                "add_date_time": (["disable", "prefix", "postfix"], {}, "Add date/time to the filename prefix. Options: disable, prefix, postfix."),
                "date_time_format": ("STRING", {"multiline": False, "default": "%Y-%m-%d_%H-%M-%S"}, "Date/time format for prefix/postfix."),
            },
            "optional": {
                "path_opt": ("STRING", {"forceInput": True}, "Optional base path to which the filename prefix will be added. If not provided, only the prefix is created."),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.FOLDER.value
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "execute"

    def execute(self, file_name_prefix, add_date_time, date_time_format, path_opt=None):
        # Joins a filename prefix (with optional date/time) to a base path, returning the new path as a string.
        # If no path is provided, returns only the prefix (with date/time if selected).
        # Handles None and empty input robustly.
        
        if not isinstance(file_name_prefix, str) or not file_name_prefix:
            file_name_prefix = "image"

        if add_date_time == "disable":
            prefix = file_name_prefix
        else:
            prefix = format_date_time(file_name_prefix, add_date_time, date_time_format)
        if path_opt and isinstance(path_opt, str) and path_opt:
            new_path = os.path.join(path_opt, prefix)  # type: ignore
        else:
            new_path = prefix
        return (new_path,)

NODE_NAME = 'Add Filename Prefix [Eclipse]'
NODE_DESC = 'Add Filename Prefix'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvFolder_FilenamePrefix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}