import json
import os
import pickle
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from urllib.parse import quote

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MODELS_FOLDER = os.path.join(BASE_DIR, "engine", "models")

LAT_LNG_API = os.getenv("LAT_LNG_API", "https://api.postcodes.io/postcodes")
CRIME_API = os.getenv("CRIME_API", "https://data.police.uk/api/crimes-street/all-crime")

DB_URL = f"postgresql+asyncpg://{os.getenv("DB_USER")}:{quote(os.getenv("DB_PASSWORD"))}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}"
DB_ENGINE = create_async_engine(
    DB_URL,
)

LINEAR_REGRESSION_MODEL = pickle.load(
    open(os.path.join(MODELS_FOLDER, "linear_reg", "linear_regression_model.pkl"), "rb")
)
LINEAR_REGRESSION_MODEL_FEATURES = json.load(
    open(os.path.join(MODELS_FOLDER, "linear_reg", "features.json"), "rb")
)
