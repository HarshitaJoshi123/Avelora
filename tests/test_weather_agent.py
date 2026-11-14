import asyncio
from pprint import pprint

from mcp_client import (
    weather_mcp_search,
    forecast_mcp_search,
)


async def main():

    city = "Mumbai"

    print("\n========== CURRENT WEATHER ==========")

    try:
        current = await weather_mcp_search(city)
        pprint(current)

    except Exception as exc:
        print("CURRENT WEATHER ERROR:", repr(exc))

    print("\n========== WEATHER FORECAST ==========")

    try:
        forecast = await forecast_mcp_search(city)
        pprint(forecast)

    except Exception as exc:
        print("FORECAST ERROR:", repr(exc))


if __name__ == "__main__":
    asyncio.run(main())