import json
import logging
import time

from openai import OpenAI, AuthenticationError, RateLimitError, APIError

from config import GROQ_API_KEY, MODEL, MAX_STEPS, LOG_LEVEL, validate_config
from tools import (
    read_file,
    write_file,
    list_files,
    run_python_code,
    web_search,
    is_search_task,
    all_definitions,
)
from memory import remember, recall_memory, log_task

# --- Startup validation ---
validate_config()

# --- Logging setup ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- API client ---
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# --- Memory tool definitions ---
memory_definitions: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save an important fact to long-term memory for future sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Retrieve all facts from long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

tools = all_definitions + memory_definitions

# --- Dispatch table: maps tool name → callable ---
# Adding a new tool requires only one entry here — no if/elif changes needed.
_TOOL_DISPATCH: dict = {
    "read_file":       lambda args: read_file(args["filename"]),
    "write_file":      lambda args: write_file(args["filename"], args["content"]),
    "list_files":      lambda args: list_files(args.get("directory", ".")),
    "run_python_code": lambda args: run_python_code(args["code"]),
    "web_search":      lambda args: web_search(args["query"]),
    "remember":        lambda args: remember(args["fact"]),
    "recall_memory":   lambda args: recall_memory(),
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Dispatch a tool call by name using the dispatch table.

    Args:
        tool_name: Name of the tool as returned by the LLM.
        tool_args: Parsed JSON arguments for the tool.

    Returns:
        String result from the tool function.
    """
    print(f"  🔧 {tool_name}({tool_args})")
    handler = _TOOL_DISPATCH.get(tool_name)
    if handler is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return f"Unknown tool: {tool_name}"
    return handler(tool_args)


def _call_llm(messages: list[dict], with_tools: bool = True) -> object:
    """
    Call the LLM API with retry logic for transient errors.

    - AuthenticationError: no retry, surfaces immediately.
    - RateLimitError: retries 3× with exponential backoff (5s, 10s, 20s).
    - APIError: retries 3× with 2s delay.

    Args:
        messages: The conversation messages list.
        with_tools: Whether to include tool definitions in the request.

    Returns:
        The API response object.

    Raises:
        AuthenticationError: If the API key is invalid.
        RateLimitError | APIError: If all retries are exhausted.
    """
    kwargs: dict = {"model": MODEL, "messages": messages}
    if with_tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            print("\n❌ Authentication failed. Check your GROQ_API_KEY in .env.")
            logger.error("Authentication error: %s", e)
            raise
        except RateLimitError as e:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
            logger.warning("Rate limit hit (attempt %d). Waiting %ds...", attempt + 1, wait)
            print(f"  ⏳ Rate limit hit. Waiting {wait}s before retry...")
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.error("API error on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

    raise RuntimeError("LLM call failed after all retries.")  # Should be unreachable


def handle_search_task(user_task: str) -> None:
    """
    Fast-path handler for tasks identified as web search requests.

    Bypasses the tool loop: directly calls web_search, then summarizes
    the results with a second LLM call.

    Args:
        user_task: The user's original task string.
    """
    print("\n📍 Step 1")
    print(f"  🔧 web_search('{user_task}')")
    search_result = web_search(user_task)

    if search_result.startswith("Search error"):
        print(f"  ❌ {search_result}")
        log_task(user_task, search_result, completed=False)
        return

    print("\n📍 Step 2 — Summarizing results...")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize search results clearly in bullet points.",
        },
        {
            "role": "user",
            "content": (
                f"User asked: {user_task}\n\n"
                f"Search results:\n{search_result}\n\n"
                "Summarize the key points."
            ),
        },
    ]

    response = _call_llm(messages, with_tools=False)
    answer = response.choices[0].message.content
    print(f"\n✅ Agent: {answer}")
    log_task(user_task, answer)


def run_agent(user_task: str) -> None:
    """
    Main agent loop. Plans and executes a task using tools until complete
    or until MAX_STEPS is reached.

    Args:
        user_task: The user's task string.
    """
    print(f"\n🎯 Task: {user_task}")
    print("─" * 50)

    if is_search_task(user_task):
        handle_search_task(user_task)
        return

    past_memory = recall_memory()

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are an efficient AI agent running on the user's local computer.\n"
                "You have tools to read/write files in the workspace, run Python code, "
                "search the web, and remember facts.\n\n"
                "STRICT RULES:\n"
                "- Only do exactly what the user asks. Nothing more.\n"
                "- Once the task is complete, stop immediately and give a short answer.\n"
                "- Do NOT call recall_memory unless the user asks what you remember.\n"
                "- Do NOT write files unless the user asks you to.\n"
                "- Do NOT run code unless the user asks you to.\n\n"
                "Your approach:\n"
                "1. PLAN — identify the minimum steps needed\n"
                "2. ACT — use only the tools required\n"
                "3. STOP — as soon as the task is done\n\n"
                f"Long-term memory from previous sessions:\n{past_memory}"
            ),
        },
        {"role": "user", "content": user_task},
    ]

    step = 1

    while step <= MAX_STEPS:
        print(f"\n📍 Step {step}")

        try:
            response = _call_llm(messages)
        except AuthenticationError:
            return  # Error message already printed in _call_llm
        except (RateLimitError, APIError) as e:
            print(f"  ❌ LLM error after retries: {e}")
            log_task(user_task, f"LLM error: {e}", completed=False)
            return

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                try:
                    tool_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    tool_args = json.loads(args_str) if args_str and args_str.strip() else {}
                    result = execute_tool(tool_name, tool_args)
                except json.JSONDecodeError as e:
                    result = f"Tool argument parse error: {e}"
                    logger.error("JSON decode error for tool '%s': %s", tool_call.function.name, e)
                except KeyError as e:
                    result = f"Missing required tool argument: {e}"
                    logger.error("Missing argument for tool '%s': %s", tool_call.function.name, e)
                except Exception as e:
                    result = f"Tool execution error: {e}"
                    logger.error(
                        "Unexpected error in tool '%s': %s",
                        tool_call.function.name, e,
                        exc_info=True,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            step += 1

        else:
            print(f"\n✅ Agent: {message.content}")
            log_task(user_task, message.content)
            break

    else:
        print("\n⚠️ Agent reached max steps without completing the task.")
        log_task(
            user_task,
            f"Incomplete — reached MAX_STEPS ({MAX_STEPS})",
            completed=False,
        )


if __name__ == "__main__":
    print("🤖 Local AI Agent — Type 'quit' to exit")
    print("─" * 50)

    while True:
        task = input("\n> What should I do? ")
        if task.lower() == "quit":
            break
        run_agent(task)
