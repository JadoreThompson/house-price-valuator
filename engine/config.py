import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MISC_FOLDER = os.path.join(os.path.dirname(BASE_DIR), "msc")
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")
RAW_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "raw-data")
CLEANED_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, "cleaned-data")
CLEANED_ZOOPLA_FOLDER = os.path.join(CLEANED_DATASETS_FOLDER, "zoopla")

for path in (
    MISC_FOLDER,
    DATASETS_FOLDER,
    RAW_DATASETS_FOLDER,
    CLEANED_DATASETS_FOLDER,
    CLEANED_ZOOPLA_FOLDER,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
