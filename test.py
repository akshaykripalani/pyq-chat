from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)  # reads API key from GOOGLE_API_KEY env var

prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
budget = 1024  # You can set this to any value between 0 and 24576

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=budget)
    ),
)

print(response.text)
