from pydantic import BaseModel
from typing import Optional

class UserIn(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class HabitsIn(BaseModel):
    exercises_per_week: int
    daily_calories: int
    alcohol_doses_per_week: float
    years_smoking: int
    hours_of_sleep: float
    stress_level: int

class UploadResponse(BaseModel):
    status: str
    id: int

class PredictResponse(BaseModel):
    prediction: float
    model: str
    explanations: dict
