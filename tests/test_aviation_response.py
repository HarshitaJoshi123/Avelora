
import asyncio
from pprint import pprint

from mcp_client import aviation_mcp_call


async def main():

    print("\nTesting AviationStack MCP...\n")

    result = await aviation_mcp_call(
        "flight_arrival_departure_schedule",
        {
            "airport_iata_code": "DEL",
            "schedule_type": "departure",
            "number_of_flights": 100,
        }
    )

    print("RESPONSE TYPE:")
    print(type(result))

    print("\nRAW MCP RESPONSE:")
    pprint(result, width=120)

    if isinstance(result, dict):
        print("\nRESPONSE KEYS:")
        print(list(result.keys()))

        if isinstance(result.get("data"), list):
            print("\nFIRST FLIGHT RECORD:")
            pprint(result["data"][:1])

    elif isinstance(result, list):
        print("\nFIRST RESPONSE ITEMS:")
        pprint(result[:2])


if __name__ == "__main__":
    asyncio.run(main())
