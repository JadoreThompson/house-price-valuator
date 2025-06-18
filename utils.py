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
        "address": address.get("address"),
        "city": address.get("city"),
        "postcode": address.get("postcode"),
        "lat": address.get("lat"),
        "lng": address.get("lng"),
        **{f"crime_{k}": v for k, v in crime.items()},
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
    parsed["batrooms"] = (
        int(bathrooms.replace("baths", "").replace("bath", "").strip())
        if bathrooms is not None
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
