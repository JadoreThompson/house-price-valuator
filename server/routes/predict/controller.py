from aiohttp import ClientSession
from collections import defaultdict
from datetime import datetime
from sqlalchemy import select
from typing import Literal
from urllib.parse import quote

from config import CRIME_API, LAT_LNG_API
from db_models import ResidentialHomes
from utils.db import get_db_session
from .models import EPCRating, PropertyType, TenureType


CRIME_CATEGORIES = [
    "violence",
    "antiSocial",
    "vehicle",
    "criminalDamage",
    "burglary",
    "publicOrder",
    "drugs",
    "theft",
    "theft",
    "theft",
    "robbery",
    "theft",
    "weaponPossession",
    "other",
]

CRIME_CATEGORY_MAP = {
    "violent-crime": "violence",
    "anti-social-behaviour": "antiSocial",
    "vehicle-crime": "vehicle",
    "criminal-damage-arson": "criminalDamage",
    "burglary": "burglary",
    "public-order": "publicOrder",
    "drugs": "drugs",
    "other-theft": "theft",
    "theft-from-the-person": "theft",
    "shoplifting": "theft",
    "robbery": "robbery",
    "bicycle-theft": "theft",
    "possession-of-weapons": "weaponPossession",
    "other-crime": "other",
}


async def fetch_lat_lng(postcode: str) -> tuple[float, float]:
    async with ClientSession() as sess:
        rsp = await sess.get(f"{LAT_LNG_API}/{quote(postcode)}")
        data = await rsp.json()
        if not data or rsp.status != 200:
            raise ValueError("Invalid postcode or API error")
        return data["result"]["latitude"], data["result"]["longitude"]


def calculate_crime_rate(crimes: list[dict]) -> dict:
    data = defaultdict(int)

    for crime in crimes:
        category = CRIME_CATEGORY_MAP[crime["category"]]
        data[f"crime_{category}"] += 1
        
    for cat in CRIME_CATEGORIES:
        if cat not in data:
            data[f"crime_{cat}"] = 0

    return data


async def fetch_crimes(lat: float, lng: float) -> dict:
    async with ClientSession() as sess:
        rsp = await sess.get(f"{CRIME_API}?lat={lat}&lng={lng}")
        data = await rsp.json()
        if not data or rsp.status != 200:
            raise ValueError("Error fetching crime data")

        return calculate_crime_rate(data)


async def get_fetaures_lr(data: dict) -> dict:
    cur_date = datetime.now()

    features = {
        "sqm_per_bed": data["sqm"] / data["bedrooms"],
        "sqm_per_room": data["sqm"] / (data["bedrooms"] + data["receptions"]),
        "total_rooms": data["bedrooms"] + data["receptions"],
        "total_crime": sum(data[key] for key in data if key.startswith("crime_")),
        "month": cur_date.month,
        "quarter": cur_date.month // 3 + 1,
    }

    for val in PropertyType:
        features[f"property_type_{val.value}"] = data["property_type"] == val.value

    for val in TenureType:
        features[f"tenure_{val.value}"] = data["tenure"] == val.value

    for val in EPCRating:
        features[f"epc_rating_{val.value}"] = data["epc_rating"] == val.value

    async with get_db_session() as sess:
        res = await sess.execute(select(ResidentialHomes.regional_avg_price).limit(1))
        features["regional_avg_price"] = res.scalar_one_or_none() or 250_000.0

    return features


async def get_features(data: dict, mtype: Literal["lr"] = "lr") -> dict:
    if mtype == "lr":
        return await get_fetaures_lr(data)
    else:
        raise ValueError(f"Unsupported model type: {mtype}")
