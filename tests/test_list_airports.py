import asyncio
from mcp_client import aviation_mcp_call


async def main():
    print("Testing AviationStack list_airports...")

    try:
        result = await aviation_mcp_call(
            "list_airports",
            {
                "limit": 5,
                "offset": 0,
                "search": "Delhi"
            }
        )

        print("\nAPI RESPONSE:")
        print(result)

    except Exception as e:
        print("\nAPI ERROR:")
        print(repr(e))


if __name__ == "__main__":
    asyncio.run(main())