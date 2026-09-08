"""Local-only, repository-scoped self-update support for Eclipse."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_VERSION_RE = re.compile(r"(?m)^version\s*=\s*['\"]([^'\"]+)['\"]")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_UPDATE_LOCK = threading.Lock()
_GIT_TIMEOUT_SECONDS = 300
_PIP_TIMEOUT_SECONDS = 1200


class SelfUpdateError(RuntimeError):
    """A sanitized, stage-specific self-update failure."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def read_disk_version(repo_root: Path) -> str:
    """Read the version currently present on disk."""
    try:
        match = _VERSION_RE.search(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8"),
        )
    except OSError:
        return "unknown"
    return match.group(1) if match else "unknown"


def get_update_status(repo_root: Path, running_version: str) -> dict[str, Any]:
    git_path = shutil.which("git")
    git_checkout = (repo_root / ".git").exists()
    disk_version = read_disk_version(repo_root)
    return {
        "success": True,
        "status": "ready" if git_path and git_checkout else "unsupported",
        "running_version": running_version,
        "disk_version": disk_version,
        "git_supported": bool(git_path and git_checkout),
        "message": (
            "Ready to update from the official repository."
            if git_path and git_checkout
            else "Self-update requires a Git checkout and the Git executable."
        ),
        "restart_required": running_version != disk_version,
    }


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    stage: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=True,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as error:
        raise SelfUpdateError(stage, f"{stage} timed out") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise SelfUpdateError(stage, f"{stage} failed") from error


def _git_raw_output(git_path: str, repo_root: Path, *arguments: str) -> str:
    result = _run_command(
        [git_path, "-C", str(repo_root), *arguments],
        cwd=repo_root,
        timeout=_GIT_TIMEOUT_SECONDS,
        stage="Git operation",
    )
    return result.stdout


def _git_output(git_path: str, repo_root: Path, *arguments: str) -> str:
    return _git_raw_output(git_path, repo_root, *arguments).strip()


def _validate_git_checkout(git_path: str, repo_root: Path) -> None:
    reported_root = _git_output(git_path, repo_root, "rev-parse", "--show-toplevel")
    try:
        matches = Path(reported_root).resolve() == repo_root.resolve()
    except OSError as error:
        raise SelfUpdateError("Git validation", "Could not resolve repository root") from error
    if not matches:
        raise SelfUpdateError("Git validation", "Git checkout root does not match this pack")


def _target_paths(git_path: str, repo_root: Path, commit: str) -> list[str]:
    output = _git_raw_output(
        git_path,
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
    )
    return [name for name in output.split("\0") if name]


def _tracked_paths(git_path: str, repo_root: Path) -> set[str]:
    output = _git_raw_output(git_path, repo_root, "ls-files", "-z")
    return {name for name in output.split("\0") if name}


def _untracked_paths(git_path: str, repo_root: Path) -> set[str]:
    visible = _git_raw_output(
        git_path,
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    ignored = _git_raw_output(
        git_path,
        repo_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    return {
        name
        for output in (visible, ignored)
        for name in output.split("\0")
        if name
    }


def _find_untracked_collision(
    git_path: str, repo_root: Path, target_commit: str,
) -> str | None:
    tracked = _tracked_paths(git_path, repo_root)
    target_paths = _target_paths(git_path, repo_root, target_commit)
    for untracked_name in _untracked_paths(git_path, repo_root):
        untracked_parts = Path(untracked_name).parts
        for target_name in target_paths:
            target_parts = Path(target_name).parts
            shared_length = min(len(untracked_parts), len(target_parts))
            if untracked_parts[:shared_length] == target_parts[:shared_length]:
                return untracked_name
    for relative_name in target_paths:
        if relative_name in tracked:
            continue
        target = repo_root / relative_name
        if target.exists() or target.is_symlink():
            return relative_name
        parent = target.parent
        while parent != repo_root:
            if parent.exists() and not parent.is_dir():
                return str(parent.relative_to(repo_root))
            parent = parent.parent
    return None


def perform_self_update(
    repo_root: Path,
    official_repository: str,
    running_version: str,
) -> dict[str, Any]:
    """Force tracked content to official main while preserving untracked data."""
    if not _UPDATE_LOCK.acquire(blocking=False):
        return {
            "success": False,
            "status": "busy",
            "error": "An update is already running.",
            "running_version": running_version,
            "disk_version": read_disk_version(repo_root),
            "restart_required": False,
        }
    try:
        git_path = shutil.which("git")
        if git_path is None or not (repo_root / ".git").exists():
            return {
                **get_update_status(repo_root, running_version),
                "success": False,
                "status": "unsupported",
                "error": "Self-update requires a Git checkout and the Git executable.",
            }

        _validate_git_checkout(git_path, repo_root)
        before_commit = _git_output(git_path, repo_root, "rev-parse", "HEAD")
        tracked_changes = bool(
            _git_output(
                git_path,
                repo_root,
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
        )
        _git_output(
            git_path,
            repo_root,
            "fetch",
            "--force",
            "--no-tags",
            official_repository,
            "refs/heads/main",
        )
        target_commit = _git_output(git_path, repo_root, "rev-parse", "FETCH_HEAD")
        if not _COMMIT_RE.fullmatch(target_commit):
            raise SelfUpdateError(
                "Git validation",
                "Official main returned an invalid commit",
            )

        collision = _find_untracked_collision(git_path, repo_root, target_commit)
        if collision is not None:
            return {
                "success": False,
                "status": "untracked_conflict",
                "error": f"Untracked user path would be overwritten: {collision}",
                "running_version": running_version,
                "disk_version": read_disk_version(repo_root),
                "before_commit": before_commit,
                "target_commit": target_commit,
                "restart_required": False,
            }

        _git_output(git_path, repo_root, "reset", "--hard", target_commit)
        changed = tracked_changes or before_commit != target_commit
        disk_version = read_disk_version(repo_root)
        dependencies_updated = False
        if before_commit != target_commit or running_version != disk_version:
            try:
                _run_command(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "-r",
                        str(repo_root / "requirements.txt"),
                    ],
                    cwd=repo_root,
                    timeout=_PIP_TIMEOUT_SECONDS,
                    stage="Dependency installation",
                )
                dependencies_updated = True
            except SelfUpdateError:
                return {
                    "success": False,
                    "status": "dependency_failed",
                    "error": "Code updated, but dependency installation failed.",
                    "running_version": running_version,
                    "disk_version": disk_version,
                    "before_commit": before_commit,
                    "after_commit": target_commit,
                    "updated": True,
                    "dependencies_updated": False,
                    "restart_required": True,
                }

        return {
            "success": True,
            "status": "updated" if changed else "up_to_date",
            "message": (
                "Update installed. Restart ComfyUI to load it."
                if changed
                else "Already on the latest official version."
            ),
            "running_version": running_version,
            "disk_version": disk_version,
            "before_commit": before_commit,
            "after_commit": target_commit,
            "updated": changed,
            "dependencies_updated": dependencies_updated,
            "restart_required": changed or running_version != disk_version,
        }
    except SelfUpdateError as error:
        return {
            "success": False,
            "status": "failed",
            "stage": error.stage,
            "error": str(error),
            "running_version": running_version,
            "disk_version": read_disk_version(repo_root),
            "restart_required": running_version != read_disk_version(repo_root),
        }
    finally:
        _UPDATE_LOCK.release()
