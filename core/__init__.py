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

from .keys import *
from .common import *

__all__ = ["keys", "common", "__version__", "version"]

# Determine package version from pyproject.toml when possible so the version
# follows packaging metadata and does not need manual edits.
from pathlib import Path


def _read_pyproject_version() -> str:
	p = Path(__file__).resolve()
	for parent in p.parents:
		candidate = parent / "pyproject.toml"
		if not candidate.exists():
			continue
		txt = candidate.read_text(encoding="utf-8")
		try:
			import tomllib as _toml  # Python 3.11+

			data = _toml.loads(txt)
		except Exception:
			try:
				import toml as _toml  # type: ignore

				data = _toml.loads(txt)
			except Exception:
				data = None

		if isinstance(data, dict):
			project = data.get("project")
			if isinstance(project, dict) and "version" in project:
				v = project.get("version")
				if isinstance(v, str):
					return v
			tool = data.get("tool")
			if isinstance(tool, dict):
				poetry = tool.get("poetry")
				if isinstance(poetry, dict) and "version" in poetry:
					v = poetry.get("version")
					if isinstance(v, str):
						return v

		# Fallback regex
		import re

		m = re.search(r"\bversion\s*=\s*['\"]([^'\"]+)['\"]", txt)
		if m:
			return m.group(1)

	# Final fallback
	# Historically the project used version "1.15.0"; use it as a safer
	# fallback when pyproject.toml can't be parsed.
	return "1.15.0"


__version__ = _read_pyproject_version()
version = __version__
