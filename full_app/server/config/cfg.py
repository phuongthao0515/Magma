import os

from dotenv import load_dotenv

load_dotenv()

ENV = os.environ.get("ENV", "development")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# In production, this would be the model API key / endpoint
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "")
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "")


def is_dev() -> bool:
    return ENV == "development"
