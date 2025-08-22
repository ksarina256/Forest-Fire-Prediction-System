from pydantic import BaseModel, Field
from typing import Literal

class FireFeatures(BaseModel):
    FFMC: float = Field(..., description="Fine Fuel Moisture Code")
    DMC: float = Field(..., description="Duff Moisture Code")
    DC: float = Field(..., description="Drought Code")
    ISI: float = Field(..., description="Initial Spread Index")
    temp: float = Field(..., description="Temperature (C)")
    RH: float = Field(..., description="Relative Humidity (%)")
    wind: float = Field(..., description="Wind speed (km/h)")
    rain: float = Field(..., description="Rain (mm/m2)")
    month: Literal["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    day: Literal["mon","tue","wed","thu","fri","sat","sun"]

class Prediction(BaseModel):
    risk: int
    probability: float
    top_features: list[str]
