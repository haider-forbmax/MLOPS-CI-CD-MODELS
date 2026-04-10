import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """
    Configuration for NIMAR Sentiment Analysis API
    """

    # General API configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Model and endpoint details
    # BASE_URL = os.getenv("BASE_URL", "http://vllm-gptoss-service.models-inference.svc.cluster.local:8000/v1")
    BASE_URL = os.getenv("BASE_URL", "https://gptoss-dgx.nimar.gov.pk/v1") #FOR DEV
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-20b")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")      # GPT-OSS model access key
    API_KEY = os.getenv("API_KEY", "123")           # internal key for client authentication

    # Logging configuration
    LOG_FILE = os.getenv("LOG_FILE", "translation_summarization.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    DYNAMIC_TRANSLATION_PROMPT = os.getenv("DYNAMIC_TRANSLATION_PROMPT", "")
    TRANSLATION_EXAMPLES = os.getenv("TRANSLATION_EXAMPLES","")
    DYNAMIC_SUMMARIZATION_PROMPT = os.getenv("DYNAMIC_SUMMARIZATION_PROMPT","")
    SUMMARIZATION_EXAMPLES = os.getenv("SUMMARIZATION_EXAMPLES","")
    # Service information
    API_TITLE = os.getenv("SERVICE_NAME", "Translation & Summarization API")
    API_VERSION = os.getenv("SERVICE_VERSION", "v1")

    # Health check and diagnostics
    HEALTH_ENDPOINT = os.getenv("HEALTH_ENDPOINT", f"{BASE_URL}/models")

    # Extra settings (for future scalability)
    RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "json")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    
    INPUT_TOKENS_LIMIT = 80000
    CONTEXT_WINDOW = 128000   