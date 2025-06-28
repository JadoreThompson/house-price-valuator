from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from enum import Enum


class PropertyType(str, Enum):
    BUNGALOW = "bungalow"
    DETACHED_BUNGALOW = "detached bungalow"
    DETACHED = "detached"
    END_TERRACE = "end terrace"
    FLAT = "flat"
    MID_TERRACE = "mid terrace"
    MID_TERRACE_BUNGALOW = "mid terrace bungalow"
    SEMI_DETACHED_BUNGALOW = "semi-detached bungalow"
    SEMI_DETACHED_HOUSE = "semi detached house"
    TERRACE_PROPERTY = "terrace property"


class TenureType(str, Enum):
    FEUDAL = "feudal"
    FREEHOLD = "freehold"
    LEASEHOLD = "leasehold"


class EPCRating(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"


class PredictRequest(BaseModel):
    class Config:
        use_enum_values = True

    sqm: int = Field(..., gt=0)
    bedrooms: int = Field(..., gt=0)
    bathrooms: int = Field(..., gt=0)
    receptions: int = Field(..., ge=0)
    postcode: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    property_type: PropertyType
    tenure: Optional[TenureType] = TenureType.FREEHOLD
    epc_rating: Optional[EPCRating] = EPCRating.D

    @model_validator(mode="before")
    def class_validator(cls, values):
        if (not values.get("lat") or not values.get("lng")) and not values.get(
            "postcode"
        ):
            raise ValueError("Either postcode or lat/lng must be provided.")

        if values.get("lat") and not values.get("lng"):
            raise ValueError("If lat is provided, lng must also be provided.")

        if values.get("lng") and not values.get("lat"):
            raise ValueError("If lng is provided, lat must also be provided.")

        return values
