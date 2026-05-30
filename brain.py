from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Only these 2 lines are different from OpenAI
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # current free model on Groq
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

print(response.choices[0].message.content)