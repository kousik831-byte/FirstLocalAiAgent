import os
import sys

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# --- Model Settings ---
MODEL: str = "llama-3.3-70b-versatile"
MAX_STEPS: int = 10

# --- Memory ---
MEMORY_FILE: str = "memory/agent_memory.json"

# --- File Safety ---
WORKSPACE_DIR: str = os.path.abspath("workspace")
MAX_FILE_SIZE_MB: int = 10

# --- Search ---
SEARCH_MAX_RESULTS: int = 5

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")


def validate_config() -> None:
    """Validate required API keys are present. Exits with a clear message if any are missing."""
    missing = [
        key for key, val in [
            ("GROQ_API_KEY", GROQ_API_KEY),
            ("TAVILY_API_KEY", TAVILY_API_KEY),
        ]
        if not val
    ]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("   Add them to your .env file and retry.")
        sys.exit(1)
