from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
import os
import json
from datetime import datetime

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ── LONG-TERM MEMORY ───────────────────────────────────────────────────

MEMORY_FILE = "agent_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"facts": [], "history": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def remember(fact):
    memory = load_memory()
    memory["facts"].append({
        "fact": fact,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)
    return f"Remembered: {fact}"

def recall_memory():
    memory = load_memory()
    if not memory["facts"]:
        return "No memories yet."
    return "\n".join([f"- {m['fact']}" for m in memory["facts"]])

def log_task(task, result):
    memory = load_memory()
    memory["history"].append({
        "task": task,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)

# ── TOOLS ──────────────────────────────────────────────────────────────

def read_file(filename):
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(filename, content):
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_files(directory="."):
    try:
        return "\n".join(os.listdir(directory))
    except Exception as e:
        return f"Error listing files: {str(e)}"

def run_python_code(code):
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout or result.stderr
        return output if output else "Code ran with no output"
    except Exception as e:
        return f"Error running code: {str(e)}"

def web_search(query):
    try:
        results = tavily.search(query=query, max_results=3)
        output = []
        for r in results["results"]:
            output.append(f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content'][:300]}\n")
        return "\n---\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

# ── TOOL DEFINITIONS ───────────────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save an important fact to long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Retrieve all facts from long-term memory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ── TOOL EXECUTOR ──────────────────────────────────────────────────────

def execute_tool(tool_name, tool_args):
    print(f"  🔧 {tool_name}({tool_args})")
    if tool_name == "read_file":
        return read_file(tool_args["filename"])
    elif tool_name == "write_file":
        return write_file(tool_args["filename"], tool_args["content"])
    elif tool_name == "list_files":
        return list_files(tool_args.get("directory", "."))
    elif tool_name == "run_python_code":
        return run_python_code(tool_args["code"])
    elif tool_name == "remember":
        return remember(tool_args["fact"])
    elif tool_name == "recall_memory":
        return recall_memory()
    return f"Unknown tool: {tool_name}"

# ── SEARCH INTENT DETECTOR ─────────────────────────────────────────────

def is_search_task(task):
    keywords = ["search", "latest", "news", "current", "today",
                "find online", "look up", "what is happening",
                "recent", "trending", "web"]
    return any(word in task.lower() for word in keywords)

# ── SEARCH + SUMMARIZE (bypasses tool calling entirely) ────────────────

def handle_search_task(user_task):
    print("\n📍 Step 1")
    print(f"  🔧 web_search('{user_task}')")

    search_result = web_search(user_task)

    if "Search error" in search_result:
        print(f"  ❌ {search_result}")
        return

    print("\n📍 Step 2 — Summarizing results...")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize the search results clearly and concisely in bullet points."
        },
        {
            "role": "user",
            "content": f"User asked: {user_task}\n\nSearch results:\n{search_result}\n\nSummarize the key points."
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    print(f"\n✅ Agent: {answer}")
    log_task(user_task, answer)

# ── THE AGENT LOOP ─────────────────────────────────────────────────────

def run_agent(user_task):
    print(f"\n🎯 Task: {user_task}")
    print("─" * 50)

    # Route search tasks directly — bypass tool calling
    if is_search_task(user_task):
        handle_search_task(user_task)
        return

    past_memory = recall_memory()

    messages = [
        {
            "role": "system",
            "content": f"""You are an efficient AI agent running on the user's local computer.
You have tools to read/write files, run Python code, and remember facts.

STRICT RULES:
- Only do exactly what the user asks. Nothing more.
- Do NOT use extra tools unless the task requires it.
- Once the task is complete, stop immediately and give a short answer.
- Do NOT call recall_memory unless the user asks what you remember.
- Do NOT write files unless the user asks you to.
- Do NOT run code unless the user asks you to.

Your approach:
1. PLAN — identify the minimum steps needed
2. ACT — use only the tools required
3. STOP — as soon as the task is done

Long-term memory from previous sessions:
{past_memory}
"""
        },
        {"role": "user", "content": user_task}
    ]

    step = 1
    max_steps = 10

    while step <= max_steps:
        print(f"\n📍 Step {step}")

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"  ⚠️ LLM error: {e}")
            print("  🔄 Retrying with simplified context...")
            messages.append({
                "role": "user",
                "content": f"Your last tool call had a formatting error: {str(e)}. Please try a simpler approach."
            })
            step += 1
            continue

        message = response.choices[0].message

        # Agent wants to use tools
        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                try:
                    tool_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    tool_args = json.loads(args_str) if args_str and args_str.strip() else {}
                    result = execute_tool(tool_name, tool_args)
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"
                    print(f"  ❌ Tool error: {e}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

            step += 1

        # Agent is done
        else:
            print(f"\n✅ Agent: {message.content}")
            log_task(user_task, message.content)
            break

    else:
        print("\n⚠️ Agent reached max steps without completing the task.")

# ── CLI INTERFACE ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Local AI Agent v3 — Type 'quit' to exit")
    print("─" * 50)

    while True:
        task = input("\n> What should I do? ")
        if task.lower() == "quit":
            break
        run_agent(task)