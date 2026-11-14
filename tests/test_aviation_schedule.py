import asyncio
from mcp_client import aviation_mcp_call


# async def main():
#     print("Testing AviationStack API...")

#     try:
#         result = await aviation_mcp_call("get_flight_status", {
#             "flight_iata": "AI101"
#         })

#         print("\nAPI RESPONSE:")
#         print(result)

#     except Exception as e:
#         print("\nAPI ERROR:")
#         print(repr(e))


# if __name__ == "__main__":
#     asyncio.run(main())

async def main():
    print("Testing AviationStack departure schedule...")

    try:
        result = await aviation_mcp_call(
            "flight_arrival_departure_schedule",
            {
                "airport_iata_code": "DEL",
                "schedule_type": "departure",
                "number_of_flights": 1
            }
        )

        print("\nAPI RESPONSE:")
        print(result)

    except Exception as e:
        print("\nAPI ERROR:")
        print(repr(e))


if __name__ == "__main__":
    asyncio.run(main())