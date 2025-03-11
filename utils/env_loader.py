import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_gemini_api_key():
    return os.getenv('GEMINI_API_KEY')

def get_huggingface_api_key():
    return os.getenv('HUGGINGFACE_API_KEY')

def get_deepseek_api_key():
    return os.getenv('DEEPSEEK_API_KEY')

def get_litellm_model_default():
    return os.getenv('LITELLM_MODEL_DEFAULT')

def get_temperature():
    return os.getenv('temperature')

# Add more functions for other environment variables as needed
