import os
from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://face_user:face_password@localhost:5432/face_db"
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "10.0"))
INSIGHTFACE_MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
PORT = int(os.getenv("PORT", "8000"))
