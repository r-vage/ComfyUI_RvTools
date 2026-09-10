import os


def normalize_relative_folder_path(path: str) -> str:
    """Convert either slash style in a relative folder path to the host separator."""
    if os.path.isabs(path):
        return path
    return path.replace("\\", os.sep).replace("/", os.sep)
