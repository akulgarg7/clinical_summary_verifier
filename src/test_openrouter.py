import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=30.0
)

MODEL = os.getenv("OPENROUTER_MODEL")

print(f"Testing model: {MODEL}")
print("Sending request...")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: API TEST OK"
        }
    ]
)

print("Response received:")
print(response.choices[0].message.content)