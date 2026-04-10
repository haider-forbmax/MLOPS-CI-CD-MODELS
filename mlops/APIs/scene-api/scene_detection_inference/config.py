import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config:
    API_KEY = os.getenv("API_KEY", "123")
    # API_URL = "https://florence-dgx.nimar.gov.pk/" # FOR DEV
    API_URL = os.getenv(
        "API_URL",
        "http://florence2-service.models-inference.svc.cluster.local:8000",
    )
