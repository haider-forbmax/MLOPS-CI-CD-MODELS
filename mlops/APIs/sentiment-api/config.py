# config.py
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
    BASE_URL = os.getenv("BASE_URL", "http://vllm-gptoss-service.models-inference.svc.cluster.local:8000/v1") #https://gptoss-dgx.nimar.gov.pk/v1
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-20b")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")      # GPT-OSS model access key
    API_KEY = os.getenv("API_KEY", "123")           # internal key for client authentication
    DYNAMIC_SENTIMENT_PROMPT = os.getenv("DYNAMIC_SENTIMENT_PROMPT", """Classify the given text into one of these sentiments **from Pakistan's perspective**:
                                                                            - Positive: Expressing satisfaction, approval, optimism, progress, or benefit for Pakistan
                                                                            - Negative: Expressing criticism, threat, harm, instability, or disadvantage for Pakistan
                                                                            - Neutral: Factual, balanced, or emotionless statements without clear impact on Pakistan""")
    EXAMPLES = os.getenv("EXAMPLES",'{"Positive":["Pakistan economy is improving","Exports are rising"],"Negative":["Policy failure caused inflation"]}')
    CLASSES = os.getenv("CLASSES",["Positive", "Negative", "Neutral", "Very Positive", "Very Negative"])

    # Logging configuration
    LOG_FILE = os.getenv("LOG_FILE", "sentiment.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_EXAMPLES_PER_CLASS = os.getenv("MAX_EXAMPLES_PER_CLASS", 5)
    MIN_LENGTH = os.getenv("MIN_LENGTH", 20)
    MAX_LENGTH = os.getenv("MAX_LENGTH", 500)
    # Service information
    SERVICE_NAME = os.getenv("SERVICE_NAME", "NIMAR Sentiment Analysis API")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "v1")

    # Health check and diagnostics
    HEALTH_ENDPOINT = os.getenv("HEALTH_ENDPOINT", f"{BASE_URL}/models")

    # Extra settings (for future scalability)
    RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "json")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
