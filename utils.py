def to_lowercase(obj: dict) -> dict:
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = v.lower()
        elif isinstance(v, dict):
            obj[k] = to_lowercase(v)
        elif isinstance(v, list):
            obj[k] = [to_lowercase(item) for item in v]
    return obj

def flatten_zoopla_data(data: dict) -> dict:
    features = data.get("features", {})
    address = data.get("address", {})
    crime = data.get("crime", {})

    sqm = features.get("sqm", "").replace("sqm", "").strip()
    epc = features.get("epcRating", "").replace("epc rating:", "").strip()
    beds = features.get("bedrooms", "").replace("beds", "").strip()

    return {
        "uprn": data.get("uprn"),
        "sold_date": data.get("soldDate"),
        "price": data.get("price"),
        "price_str": data.get("priceStr"),
        
        "tenure": features.get("tenure"),
        "sqm": int(sqm) if sqm.isdigit() else None,
        "epc_rating": epc,
        "property_type": features.get("propertyType"),
        "bedrooms": int(beds) if beds.isdigit() else None,
        
        "address": address.get("address"),
        "city": address.get("city"),
        "postcode": address.get("postcode"),
        "lat": address.get("lat"),
        "lng": address.get("lng"),

        **{f"crime_{k}": v for k, v in crime.items()}
    }