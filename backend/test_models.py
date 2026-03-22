import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv('.env')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

with open('backend/models_out.txt', 'w', encoding='utf-8') as f:
    for m in genai.list_models():
        f.write(f"{m.name} - generateContent: {'generateContent' in m.supported_generation_methods}\\n")
