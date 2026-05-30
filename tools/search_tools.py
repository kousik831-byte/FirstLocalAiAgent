import time
import logging

from tavily import TavilyClient

from config import TAVILY_API_KEY, SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Keywords that strongly indicate the user wants a web search
_SEARCH_KEYWORDS: list[str] = [
    "search",
    "latest",
    "news",
    "today",
    "find online",
    "look up",
    "recent",
    "trending",
    "web",
    "who is",
    "what is",
    "price of",
    "weather",
]

# Keywords that indicate a local file/code operation — override search routing
_LOCAL_OP_KEYWORDS: list[str] = [
    "file",
    "folder",
    "directory",
    "write",
    "read",
    "create",
    "run",
    "execute",
    "code",
    "script",
    "workspace",
    "list",
    "open",
    "save",
    "delete",
]


def is_search_task(task: str) -> bool:
    """
    Determine whether a task should be routed to the web search path.

    Returns True only when the task contains a search keyword AND does NOT
    contain a local-operation keyword. This prevents false positives like
    "list files in the current directory" from triggering a web search.

    Args:
        task: The user's raw task string.

    Returns:
        True if web search routing is appropriate.
    """
    lower = task.lower()
    has_search_signal = any(kw in lower for kw in _SEARCH_KEYWORDS)
    has_local_signal = any(kw in lower for kw in _LOCAL_OP_KEYWORDS)
    return has_search_signal and not has_local_signal


def web_search(query: str, retries: int = 2) -> str:
    """
    Perform a web search using the Tavily API and return formatted results.

    Retries on failure with exponential backoff (1s, 2s).

    Args:
        query: Search query string.
        retries: Number of retry attempts on failure. Defaults to 2.

    Returns:
        Formatted search results as a string, or an error message string.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            results = tavily.search(query=query, max_results=SEARCH_MAX_RESULTS)
            output = []
            for r in results.get("results", []):
                output.append(
                    f"Title: {r['title']}\n"
                    f"URL: {r['url']}\n"
                    f"Summary: {r['content'][:300]}\n"
                )
            if not output:
                return "No results found."
            return "\n---\n".join(output)
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning(
                    "Search attempt %d failed (%s). Retrying in %ds...",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)
    return f"Search error after {retries + 1} attempts: {last_error}"


definitions = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, facts, or any topic "
                "that requires up-to-date data not available locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]
