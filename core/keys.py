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

from enum import Enum
from typing import Iterable

__all__ = ["TEXTS", "CATEGORY", "KEYS", "category_display"]


class TEXTS(Enum):
    CUSTOM_NODE_NAME = "Eclipse"
    LOGGER_PREFIX = "Eclipse"
    CONCAT = "concatenated"
    INACTIVE_MSG = "inactive"
    INVALID_METADATA_MSG = "Invalid metadata raw"
    FILE_NOT_FOUND = "File not found!"


class CATEGORY(Enum):
    MAIN = "🌒 Eclipse"
    LOADER = "/ Loader"
    CONVERSION = "/ Conversion"
    FOLDER = "/ Folder"
    IMAGE = "/ Image"
    PIPE = "/ Pipe"
    PRIMITIVE = "/ Primitives"
    ROUTER = "/ Router"    
    SETTINGS = "/ Settings"
    TOOLS = "/ Tools"
    TEXT = "/ Text"
    VIDEO = "/ Video"


def category_display(cat: "CATEGORY") -> str:
    # Return a cleaned, human-friendly string for a CATEGORY value.
    #
    # This strips any leading slashes and surrounding whitespace so it can be
    # shown in UIs without duplicated separators.
    # type: ignore[name-defined]
    return cat.value.lstrip("/ ").strip()


# remember, all keys should be in lowercase!
class KEYS(Enum):
    LIST = "list_string"
    PREFIX = "prefix"


# Sanity check: ensure KEYS values are lowercase to match downstream usage.
def _assert_keys_lowercase(items: Iterable[KEYS]) -> None:
    for k in items:
        if k.value != k.value.lower():
            raise AssertionError(f"KEYS value must be lowercase: {k!r}")


_assert_keys_lowercase(KEYS)
