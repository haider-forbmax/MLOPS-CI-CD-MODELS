import os
import socket
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the Document OCR API."""


    MODEL_NAME = os.getenv("MODEL_NAME", "olmocr")
    MODEL_URL = os.getenv("MODEL_URL", 'https://ocr-ft-dgx.nimar.gov.pk')

    # === Storage Paths ===
    UPLOAD_DIR = "uploads"
    BASE_FOLDER = os.path.join(UPLOAD_DIR, "temp")
    SUBFOLDERS = ["temp_image", "temp_pdf"]
    CROPS = os.path.join(BASE_FOLDER, "crops")

    API_KEY = os.getenv("API_KEY", "123")
    # USER_API_KEYS = os.getenv("API_KEY", "123")


    # === File Limits ===
    MAX_FILE_SIZE_MB = os.getenv("MAX_FILE_SIZE_MB", 10)
    MAX_IMAGE_WIDTH = os.getenv("MAX_IMAGE_WIDTH", 2000)
    MAX_IMAGE_HEIGHT = os.getenv("MAX_IMAGE_HEIGHT", 2000)
    MAX_PDF_PAGES = os.getenv("MAX_PDF_PAGES", 20)
    MAX_EXCEL_SHEETS = os.getenv("MAX_EXCEL_SHEETS", 10)
    MAX_POWERPOINT_SLIDES = os.getenv("MAX_POWERPOINT_SLIDES", 50)


    SUPPORTED_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".heic",
    ".pdf",
    ".ppt",
    ".docx", ".rtf",
    ".pptx",
    ".xlsx", ".csv",
    ".odt", ".ods",
    ".pages", ".key", ".numbers",
    ".epub",
    ".djvu", ".ps", ".eps",
    ".html", ".htm", ".txt"
    ]


    # === API and Auth ===
    # URDU_API_BASE_URL = get_ip_address()
    # API_KEY = os.getenv("API_KEY", "apikey1")

    USER_API_KEYS = dict(
        item.split(":", 1)
        for item in os.getenv("USER_API_KEYS", "").split(",")
        if ":" in item
    )

    # === Logging ===
    LOG_FILE = os.getenv("LOG_FILE", "upload.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

