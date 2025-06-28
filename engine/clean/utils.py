import json
import os
import pandas as pd
from engine.config import CLEANED_ZOOPLA_FOLDER


def lowercase_values(obj: dict) -> dict:
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = v.lower()
        elif isinstance(v, dict):
            obj[k] = lowercase_values(v)
        elif isinstance(v, list):
            obj[k] = [lowercase_values(item) for item in v]
    return obj


def strip_values(obj: dict) -> dict:
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = v.strip()
        elif isinstance(v, dict):
            obj[k] = lowercase_values(v)
        elif isinstance(v, list):
            obj[k] = [lowercase_values(item) for item in v]
    return obj


def flatten_pois(pois: list[dict]) -> dict:
    schools = [
        p for p in pois if "category" in p and "school" in (p["category"] or "").lower()
    ]
    all_distances = [p["distanceMiles"] for p in pois]
    school_distances = [p["distanceMiles"] for p in schools]

    closest_school = min(schools, key=lambda p: p["distanceMiles"], default={})
    closest_poi = min(pois, key=lambda p: p["distanceMiles"])

    return {
        "num_pois": len(pois),
        "num_schools": len(schools),
        "avg_distance_all": sum(all_distances) / len(all_distances),
        "min_distance_school": min(school_distances) if school_distances else None,
        "closest_school_gender": closest_school.get("gender"),
        "closest_school_name": closest_school.get("name"),
        "closest_poi_name": closest_poi.get("name"),
    }


def flatten_zoopla_data(data: dict) -> dict:
    features = data.get("features", {})
    address = data.get("address", {})
    crime = data.get("crime", {})

    return {
        "uprn": data.get("uprn"),
        "sold_date": data.get("soldDate"),
        "price": data.get("price"),
        "price_str": data.get("priceStr"),
        "tenure": features.get("tenure"),
        "sqm": features.get("sqm"),
        "epc_rating": features.get("epcRating"),
        "property_type": features.get("propertyType"),
        "bedrooms": features.get("bedrooms"),
        "bathrooms": features.get("bathrooms"),
        "receptions": features.get("receptions", "0 receptions"),
        "address": address.get("address"),
        "city": address.get("city"),
        "postcode": address.get("postcode"),
        "lat": address.get("lat"),
        "lng": address.get("lng"),
        **{f"crime_{k}": v for k, v in crime.items()},
        **flatten_pois(data.get("nearbyPOIs", [])),
    }


def parse_zoopla_data(data: dict) -> dict:
    parsed = data.copy()

    for key in ("tenure", "epc_rating", "property_type", "address", "city", "postcode"):
        if parsed[key]:
            parsed[key] = parsed[key].lower()

    sqm = parsed["sqm"]
    parsed["sqm"] = int(sqm.replace("sqm", "").strip()) if sqm is not None else None

    epc_rating = parsed["epc_rating"]
    parsed["epc_rating"] = (
        epc_rating.replace("epc rating:", "").strip()
        if epc_rating is not None
        else None
    )

    bedrooms = parsed["bedrooms"]
    parsed["bedrooms"] = (
        int(bedrooms.replace("beds", "").replace("bed", "").strip())
        if bedrooms is not None
        else None
    )

    bathrooms = parsed["bathrooms"]
    parsed["bathrooms"] = (
        int(bathrooms.replace("baths", "").replace("bath", "").strip())
        if bathrooms is not None
        else None
    )

    receptions = parsed["receptions"]
    parsed["receptions"] = (
        int(receptions.replace("receptions", "").replace("reception", "").strip())
        if receptions is not None
        else None
    )

    address = parsed["address"]
    end = len(address) - 1

    postcode_start = None
    for i in range(end, 0, -1):
        if address[i] == ",":
            postcode_start = i
            break

    city_start = None
    for i in range(postcode_start - 1, 0, -1):
        if address[i] == ",":
            city_start = i
            break

    parsed["street"] = address[:city_start].replace(",", "").strip()
    parsed["city"] = address[city_start:postcode_start].replace(",", "").strip()
    parsed["postcode"] = address[postcode_start:].replace(",", "").strip()

    if parsed["property_type"] is not None:
        parsed["property_type"] = (
            "flat" if "flat" in parsed["property_type"] else parsed["property_type"]
        )

    return parsed


def build_df() -> pd.DataFrame:
    jsons: list[dict] = []
    for fname in os.listdir(CLEANED_ZOOPLA_FOLDER):
        jsons.append(json.load(open(os.path.join(CLEANED_ZOOPLA_FOLDER, fname), "rb")))

    return pd.DataFrame(jsons)


# build_df().to_csv("./datasets/cleaned.csv", index=False)
