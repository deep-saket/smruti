from common import MCPClientComponent
from models.ModelManager import ModelManager

class WeatherClient(MCPClientComponent):
    """
    Client for the /weather endpoint.
    """

    async def _process_prompt(self, prompt, *args, **kwargs):
        cities = ModelManager.ner.infer(prompt, ["city"])

        weather_info = []
        for city in cities:
            weather_info.append(str(self.request(city=city)))

        return weather_info
