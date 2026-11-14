import asyncio
import json

from mcp_client import aviation_mcp_call


async def main():

    print("Testing AviationStack Future Schedule MCP...")

    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            {
                "airport_iata_code": "DEL",
                "schedule_type": "departure",
                "date": "2026-11-15",
                "number_of_flights": 5
            }
        )

        print("\nRESULT:")
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__, str(e))


if __name__ == "__main__":
    asyncio.run(main())