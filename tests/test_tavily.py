import asyncio
from pprint import pprint

from mcp_client import tavily_mcp_search


async def main():
    query = (
        "Best hotels in Mumbai, India for a 3-day trip. "
        "Include hotel names, locations, prices, amenities "
        "and official booking links."
    )

    try:
        result = await tavily_mcp_search(query)

        print("\n========== TAVILY RESPONSE ==========")
        print("Response type:", type(result))

        print("\n========== RAW RESPONSE ==========")
        pprint(result, width=120)

    except Exception as exc:
        print(
            f"TAVILY ERROR: {type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    asyncio.run(main())