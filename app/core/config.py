import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 'app' directory (parent of 'core')
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chinook.db"

class Settings:
    DB_PATH: str = str(DB_PATH)

settings = Settings()