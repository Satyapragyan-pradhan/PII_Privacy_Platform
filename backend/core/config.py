import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME = os.getenv(
        "APP_NAME",
        "PII Privacy Intelligence"
    )

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b"
    )

    MAX_FILE_SIZE_MB = int(
        os.getenv("MAX_FILE_SIZE_MB", "20")
    )

    TESSERACT_CMD = os.getenv(
        "TESSERACT_CMD",
        ""
    )


settings = Settings()