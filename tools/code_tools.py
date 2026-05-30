import subprocess
import logging

logger = logging.getLogger(__name__)

# ============================================================
# SECURITY WARNING: run_python_code executes arbitrary Python
# code as a subprocess with NO sandboxing beyond a timeout.
# This is intentional for demo/interview purposes to illustrate
# agentic code execution capability. In production this would
# require a sandbox (e.g. Docker, RestrictedPython, or a
# dedicated code execution service). Never expose this to
# untrusted input in a real deployment.
# ============================================================


def run_python_code(code: str) -> str:
    """
    Execute a Python code snippet in a subprocess and return its output.

    Stdout and stderr are returned separately so the caller can distinguish
    successful output from error messages.

    Args:
        code: Python source code to execute.

    Returns:
        Output string containing stdout, stderr (if any), or an error message.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,  # Hard timeout; no CPU/memory limits — demo only
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        return "\n".join(parts) if parts else "Code ran with no output."
    except subprocess.TimeoutExpired:
        logger.warning("Code execution timed out.")
        return "Error: Code execution timed out after 10 seconds."
    except FileNotFoundError:
        return "Error: Python interpreter not found. Is Python on PATH?"
    except OSError as e:
        logger.error("OS error running code: %s", e)
        return f"Error running code: {e}"


definitions = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": (
                "Execute a Python code snippet and return its output. "
                "Use for calculations, data processing, or generating content programmatically. "
                "Output from print() statements is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python source code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]
