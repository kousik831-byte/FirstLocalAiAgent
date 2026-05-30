import os
import logging

from config import WORKSPACE_DIR, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)

# Ensure workspace directory exists at import time
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def _safe_path(filename: str) -> str:
    """
    Resolve filename to an absolute path inside WORKSPACE_DIR.

    Uses os.path.realpath (resolves symlinks) before the prefix check —
    the only correct way to block path traversal attacks.

    Args:
        filename: Relative filename supplied by the caller (or LLM).

    Returns:
        Absolute path guaranteed to be inside WORKSPACE_DIR.

    Raises:
        ValueError: If the resolved path escapes the workspace.
    """
    clean = os.path.normpath(filename).lstrip(os.sep).lstrip(".")
    resolved = os.path.realpath(os.path.join(WORKSPACE_DIR, clean))
    if not resolved.startswith(os.path.realpath(WORKSPACE_DIR)):
        raise ValueError(
            f"Path traversal blocked: '{filename}' resolves outside workspace."
        )
    return resolved


def read_file(filename: str) -> str:
    """
    Read and return the text contents of a file inside the workspace directory.

    Args:
        filename: Relative filename within the workspace (e.g. 'notes.txt').

    Returns:
        File contents as a string, or an error message string.
    """
    try:
        path = _safe_path(filename)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return f"Error: File exceeds {MAX_FILE_SIZE_MB}MB limit ({size_mb:.1f}MB)."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except ValueError as e:
        logger.warning("Path traversal attempt blocked: %s", e)
        return f"Security error: {e}"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found in workspace."
    except PermissionError:
        return f"Error: Permission denied reading '{filename}'."
    except OSError as e:
        logger.error("OS error reading file '%s': %s", filename, e)
        return f"Error reading file: {e}"


def write_file(filename: str, content: str) -> str:
    """
    Write text content to a file inside the workspace directory.

    Creates parent directories as needed. Only writes inside workspace/.

    Args:
        filename: Relative filename within the workspace.
        content: Text content to write.

    Returns:
        Success or error message string.
    """
    try:
        path = _safe_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to workspace/{filename}"
    except ValueError as e:
        logger.warning("Path traversal attempt blocked: %s", e)
        return f"Security error: {e}"
    except PermissionError:
        return f"Error: Permission denied writing '{filename}'."
    except OSError as e:
        logger.error("OS error writing file '%s': %s", filename, e)
        return f"Error writing file: {e}"


def list_files(directory: str = ".") -> str:
    """
    List files and folders in the workspace directory (or a subdirectory of it).

    Args:
        directory: Subdirectory within workspace to list. Defaults to workspace root.

    Returns:
        Sorted, newline-separated file list, or an error message string.
    """
    try:
        path = WORKSPACE_DIR if directory in (".", "") else _safe_path(directory)
        entries = os.listdir(path)
        if not entries:
            return "No files found in workspace."
        return "\n".join(sorted(entries))
    except ValueError as e:
        logger.warning("Path traversal attempt blocked: %s", e)
        return f"Security error: {e}"
    except FileNotFoundError:
        return f"Error: Directory '{directory}' not found in workspace."
    except PermissionError:
        return f"Error: Permission denied listing '{directory}'."
    except OSError as e:
        logger.error("OS error listing directory '%s': %s", directory, e)
        return f"Error listing directory: {e}"


definitions = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file from the workspace directory. "
                "Only files inside the workspace/ folder can be accessed. "
                "Pass the filename relative to workspace (e.g. 'notes.txt')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative filename within workspace (e.g. 'notes.txt').",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file in the workspace directory. "
                "Creates the file if it does not exist. Only writes inside workspace/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative filename within workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List all files and folders in the workspace directory. "
                "Only lists inside workspace/. Leave directory empty to list the workspace root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Subdirectory within workspace to list. "
                            "Omit or pass '.' for the workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]
