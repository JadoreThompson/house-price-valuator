import os
from dotenv import load_dotenv

load_dotenv()

DATASETS_FOLDER = os.path.join(os.path.dirname(__file__), "datasets")
RAW_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "raw-data")
CLEANED_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "cleaned-data")
CLEANED_ZOOPLA_FOLDER = os.path.join(CLEANED_DATASETS_FOLDER, "zoopla")

for path in (
    DATASETS_FOLDER,
    RAW_DATASETS_FOLDER,
    CLEANED_DATASETS_FOLDER,
    CLEANED_ZOOPLA_FOLDER,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
