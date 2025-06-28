import pandas as pd
from fastapi import APIRouter

from config import LINEAR_REGRESSION_MODEL
from .controller import fetch_crimes, fetch_lat_lng, get_features
from .models import PredictRequest, PredictResponse

route = APIRouter(prefix="/predict", tags=["predict"])


@route.post("/")
async def predict(body: PredictRequest):
    body: dict = body.model_dump()

    lat, lng = await fetch_lat_lng(body["postcode"])
    crime_data = await fetch_crimes(lat, lng)
    features = await get_features(body)

    body["lat"] = lat
    body["lng"] = lng
    body.update(features)
    body.update(crime_data)

    # pred = LINEAR_REGRESSION_MODEL.predict(pd.DataFrame([body]))
    pred = 250_000.0

    return PredictResponse(
        price=pred,
        score=abs(pred - body["price"]) / body["price"] if body["price"] else None,
    )
