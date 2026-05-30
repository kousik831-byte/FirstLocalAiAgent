import json
import os
import logging
from datetime import datetime
from typing import Optional

from config import MEMORY_FILE

logger = logging.getLogger(__name__)

_SCHEMA_KEYS = {"facts", "history"}
_FACT_KEYS = {"fact", "timestamp"}
_HISTORY_KEYS = {"task", "result", "timestamp"}


def _validate_memory(data: dict) -> dict:
    """
    Validate and repair the memory dict structure.

    Missing top-level keys are added with empty lists. Entries with missing
    sub-keys are logged and dropped so a partially corrupt file degrades
    gracefully rather than crashing.

    Args:
        data: Raw dict loaded from JSON.

    Returns:
        Validated (and repaired if needed) memory dict.
    """
    if not isinstance(data, dict):
        logger.warning("Memory file had invalid root type; resetting.")
        return {"facts": [], "history": []}

    for key in _SCHEMA_KEYS:
        if key not in data or not isinstance(data[key], list):
            logger.warning("Memory key '%s' missing or invalid; resetting to [].", key)
            data[key] = []

    valid_facts = []
    for entry in data["facts"]:
        if isinstance(entry, dict) and _FACT_KEYS.issubset(entry.keys()):
            valid_facts.append(entry)
        else:
            logger.warning("Skipping malformed fact entry: %s", entry)
    data["facts"] = valid_facts

    valid_history = []
    for entry in data["history"]:
        if isinstance(entry, dict) and _HISTORY_KEYS.issubset(entry.keys()):
            valid_history.append(entry)
        else:
            logger.warning("Skipping malformed history entry: %s", entry)
    data["history"] = valid_history

    return data


def load_memory() -> dict:
    """
    Load the memory JSON file from disk.

    Returns a default empty structure if the file does not exist.
    On JSON decode errors, backs up the corrupt file and returns empty structure.

    Returns:
        Dict with keys 'facts' (list) and 'history' (list).
    """
    if not os.path.exists(MEMORY_FILE):
        return {"facts": [], "history": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _validate_memory(raw)
    except json.JSONDecodeError as e:
        logger.error("Memory file is corrupt (%s); backing up and resetting.", e)
        backup_path = MEMORY_FILE + ".corrupt"
        try:
            os.replace(MEMORY_FILE, backup_path)
            logger.info("Corrupt memory backed up to %s", backup_path)
        except OSError:
            pass
        return {"facts": [], "history": []}
    except OSError as e:
        logger.error("Could not read memory file: %s", e)
        return {"facts": [], "history": []}


def save_memory(memory: dict) -> None:
    """
    Persist the memory dict to disk as JSON.

    Uses a write-to-temp-then-rename pattern (os.replace) to ensure the file
    is never left in a partially written state if the process is interrupted.

    Args:
        memory: The memory dict to save.
    """
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    tmp_path = MEMORY_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        os.replace(tmp_path, MEMORY_FILE)  # Atomic on most OS/filesystems
    except OSError as e:
        logger.error("Failed to save memory: %s", e)


def remember(fact: str) -> str:
    """
    Append a new fact to long-term memory.

    Args:
        fact: The fact string to remember.

    Returns:
        Confirmation message string.
    """
    memory = load_memory()
    memory["facts"].append({
        "fact": fact,
        "timestamp": datetime.now().isoformat(),
    })
    save_memory(memory)
    return f"Remembered: {fact}"


def recall_memory(limit: Optional[int] = None) -> str:
    """
    Retrieve facts from long-term memory.

    Args:
        limit: If provided, return only the most recent N facts.

    Returns:
        Newline-separated list of facts, or a 'no memories' message.
    """
    memory = load_memory()
    facts = memory["facts"]
    if not facts:
        return "No memories yet."
    if limit is not None and limit > 0:
        facts = facts[-limit:]
    return "\n".join([f"- {m['fact']}" for m in facts])


def log_task(task: str, result: str, completed: bool = True) -> None:
    """
    Append a task result to history.

    Args:
        task: The original user task string.
        result: The agent's result or final message.
        completed: False if the agent hit MAX_STEPS without finishing.
    """
    memory = load_memory()
    memory["history"].append({
        "task": task,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "completed": completed,
    })
    save_memory(memory)


def delete_memory(index: int) -> str:
    """
    Delete a fact from memory by its zero-based index.

    Args:
        index: Zero-based index into the facts list.

    Returns:
        Confirmation or error message string.
    """
    memory = load_memory()
    facts = memory["facts"]
    if not (0 <= index < len(facts)):
        return f"Error: Index {index} is out of range (0–{len(facts) - 1})."
    removed = facts.pop(index)
    save_memory(memory)
    return f"Deleted memory: {removed['fact']}"


def clear_history() -> str:
    """
    Clear all task history entries while preserving facts.

    Returns:
        Confirmation message string.
    """
    memory = load_memory()
    count = len(memory["history"])
    memory["history"] = []
    save_memory(memory)
    return f"Cleared {count} history entries."
