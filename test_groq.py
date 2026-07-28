import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print("API Key Loaded:", os.getenv("GROQ_API_KEY") is not None)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=10,
    )

    print("\nSUCCESS")
    print(response.choices[0].message.content)

except Exception:
    import traceback
    traceback.print_exc()