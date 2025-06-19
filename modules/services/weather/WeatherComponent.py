from common import CallableComponent
from config.loader import settings
import requests
from schemas.mcp import WeatherInfo

class WeatherComponent(CallableComponent):
    """
    Fetch current weather for a given city using OpenWeatherMap.
    """
    def __init__(self):
        super().__init__()
        cfg = settings["mcp"]["weather"]
        self.url     = cfg["base_url"]
        self.api_key = cfg["api_key"]
        if not self.api_key:
            raise RuntimeError("WEATHER_API_KEY not set in your configuration")

    def __call__(self, city: str) -> WeatherInfo:
        params = {"q": city, "appid": self.api_key, "units": "metric"}
        resp = requests.get(self.url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        info = {
            "city":          data["name"],
            "description":   data["weather"][0]["description"],
            "temperature_c": data["main"]["temp"],
            "humidity":      data["main"]["humidity"],
        }
        return WeatherInfo(**info)

if __name__ == "__main__":
    # Simple local test
    component = WeatherComponent()
    test_city = "London"
    try:
        weather = component(test_city)
        print(f"Weather in {weather.city}:")
        print(f"  Description: {weather.description}")
        print(f"  Temperature: {weather.temperature_c} °C")
        print(f"  Humidity:    {weather.humidity}%")
    except Exception as e:
        print(f"Error fetching weather: {e}")