import os
from dotenv import load_dotenv

load_dotenv()

DATASETS_FOLDER = os.path.join(os.path.dirname(__file__), "datasets")
RAW_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "raw-data")
CLEANED_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "cleaned-data")
