from fastapi import APIRouter
from .controller import fetch_crimes, fetch_lat_lng, get_features
from .models import PredictRequest

route = APIRouter(prefix="/predict", tags=["predict"])


@route.post("/")
async def predict(body: PredictRequest):
    body: dict = body.model_dump()

    lat, lng = await fetch_lat_lng(body["postcode"])
    crime_data = await fetch_crimes(lat, lng)
    features = get_features(body)

    body["lat"] = lat
    body["lng"] = lng
    body.update(features)
    body.update(crime_data)

    return body
