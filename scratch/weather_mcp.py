from mcp_util.clients import WeatherClient

async def main2():
    client = WeatherClient()
    city = "Bhubaneswar"
    try:
        # call the client’s .req method to fetch weather
        info = await client.request(city=city)
        print(f"Weather for {city}:")
        print(f"  City:           {info['city']}")
        print(f"  Description:    {info['description']}")
        print(f"  Temperature °C: {info['temperature_c']}")
        print(f"  Humidity %:     {info['humidity']}")
    except Exception as e:
        print(f"Error fetching weather: {e}")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main2())