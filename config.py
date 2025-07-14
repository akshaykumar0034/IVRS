# config.py
import os

PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
MODEL_PATH = r"E:\ANPD\model\best.pt"
PLATE_PATTERN = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
CONFIDENCE_THRESHOLD = 0.4
