from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── TOOLS (the agent's hands) ──────────────────────────────────────────

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
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

# ── TOOL DEFINITIONS (what the LLM sees) ───────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The file path to read"}
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
                    "filename": {"type": "string", "description": "The file path to write to"},
                    "content": {"type": "string", "description": "The content to write"}
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
                    "directory": {"type": "string", "description": "The directory path to list"}
                },
                "required": []
            }
        }
    }
]

# ── TOOL EXECUTOR (your Python runs the actual tool) ───────────────────

def execute_tool(tool_name, tool_args):
    print(f"\n🔧 Executing tool: {tool_name} with args: {tool_args}")
    if tool_name == "read_file":
        return read_file(tool_args["filename"])
    elif tool_name == "write_file":
        return write_file(tool_args["filename"], tool_args["content"])
    elif tool_name == "list_files":
        return list_files(tool_args.get("directory", "."))
    else:
        return f"Unknown tool: {tool_name}"

# ── THE AGENT LOOP (think → act → observe → repeat) ────────────────────

def run_agent(user_task):
    print(f"\n🎯 Task: {user_task}")
    print("─" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to complete tasks."},
        {"role": "user", "content": user_task}
    ]

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # ── Case 1: Agent wants to use a tool ──
        if message.tool_calls:
            messages.append(message)  # add agent's decision to memory

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = execute_tool(tool_name, tool_args)

                # Send the tool result back to the agent
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # ── Case 2: Agent is done, returns final answer ──
        else:
            print(f"\n✅ Agent: {message.content}")
            break

# ── RUN IT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent("List all files in the current directory, then create a file called hello.txt with a short poem inside it.")