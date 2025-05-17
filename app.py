import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-pro')

try:
    for i in range(5):  # Example of making multiple requests
        response = model.generate_content(f"Explain a simple concept {i+1}.")
        print(f"Response {i+1}: {response.text}")
        time.sleep(5)  # Wait for 5 seconds between requests
except Exception as e:
    print(f"An error occurred: {e}")