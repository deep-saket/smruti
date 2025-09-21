from pydantic import BaseModel

class WeatherInfo(BaseModel):
    city: str
    description: str
    temperature_c: float
    humidity: int