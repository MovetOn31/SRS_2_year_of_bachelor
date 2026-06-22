import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")

VOICES_DIR = BASE_DIR / "data" / "voices"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

VOICES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOICE_DURATION_SECONDS = 60

TARGET_SAMPLE_RATE = 22050
TARGET_DURATION_SECONDS = 4


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is not set. Add BOT_TOKEN to .env file."
    )
