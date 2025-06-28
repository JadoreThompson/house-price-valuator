from aiohttp import ClientSession
from collections import defaultdict
from datetime import datetime
from urllib.parse import quote

from config import CRIME_API, LAT_LNG_API
from .models import EPCRating, PropertyType, TenureType


async def fetch_lat_lng(postcode: str) -> tuple[float, float]:
    async with ClientSession() as sess:
        rsp = await sess.get(f"{LAT_LNG_API}/{quote(postcode)}")
        data = await rsp.json()
        if not data or rsp.status != 200:
            raise ValueError("Invalid postcode or API error")
        return data["result"]["latitude"], data["result"]["longitude"]


def parse_crime_category(value: str) -> str:
    return {
        "violent-crime": "violence",
        "anti-social-behaviour": "anti_cocial",
        "vehicle-crime": "vehicle",
        "criminal-damage-arson": "criminal_damage",
        "burglary": "burglary",
        "public-order": "public_order",
        "drugs": "drugs",
        "other-theft": "theft",
        "theft-from-the-person": "theft",
        "shoplifting": "theft",
        "robbery": "robbery",
        "bicycle-theft": "theft",
        "possession-of-weapons": "weapon_possession",
        "other-crime": "other",
    }[value]


def calculate_crime_rate(crimes: list[dict]) -> dict:
    data = defaultdict(int)

    for crime in crimes:
        category = parse_crime_category(crime["category"])
        data[f"crime_{category}"] += 1

    return data


async def fetch_crimes(lat: float, lng: float) -> dict:
    async with ClientSession() as sess:
        rsp = await sess.get(f"{CRIME_API}?lat={lat}&lng={lng}")
        data = await rsp.json()
        if not data or rsp.status != 200:
            raise ValueError("Error fetching crime data")

        return calculate_crime_rate(data)


def get_features(data: dict) -> dict:
    features = {
        "sqm_per_bed": data["sqm"] / data["bedrooms"],
        "sqm_per_room": data["sqm"] / (data["bedrooms"] + data["receptions"]),
        "total_crime": sum(data[key] for key in data if key.startswith("crime_")),
    }
    
    
    for val in PropertyType:
        features[f"property_type_{val.value}"] = data['property_type'] == val.value
        
    for val in TenureType:
        features[f"tenure_{val.value}"] = data['tenure'] == val.value
        
    for val in EPCRating:
        features[f"epc_rating_{val.value}"] = data['epc_rating'] == val.value
        
    cur_date = datetime.now()
    features['month'] = cur_date.month
    features['quarter'] = cur_date.month // 3 + 1
    
    return features