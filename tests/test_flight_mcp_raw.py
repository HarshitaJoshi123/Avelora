# test_flight_mcp_raw.py
import asyncio
from pprint import pprint

from mcp_client import aviation_mcp_call


async def diag_call(leg_label, airport_code, date):
    print("\n" + "=" * 70)
    print(f"{leg_label}: airport={airport_code}, date={date}")
    print("=" * 70)

    arguments = {
        "airport_iata_code": airport_code,
        "schedule_type": "departure",
        "date": date,
        "number_of_flights": 100,
    }
    print("Arguments:")
    pprint(arguments)

    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            arguments,
        )
    except Exception as exc:
        print("\n!!! EXCEPTION CAUGHT !!!")
        print(f"  Type   : {type(exc).__name__}")
        print(f"  Message: {exc}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nReturn type: {type(result).__name__}")

    if isinstance(result, list):
        print(f"Content blocks: {len(result)}")
        for i, item in enumerate(result):
            print(f"\n--- block [{i}] type={type(item).__name__} ---")
            if isinstance(item, dict):
                print(f"  keys: {list(item.keys())}")
                text = item.get("text", "")
                print(f"  text length: {len(str(text))}")
                print(f"  text[:600]: {str(text)[:600]}")
            else:
                print(f"  raw value: {str(item)[:600]}")

    elif isinstance(result, dict):
        print(f"Dict keys: {list(result.keys())}")
        pprint(result, width=120)

    elif isinstance(result, str):
        print(f"String length: {len(result)}")
        print(f"First 600 chars: {result[:600]}")

    else:
        print(f"Value: {str(result)[:600]}")


async def main():
    await diag_call("OUTBOUND", "DEL", "2026-11-20")

    print("\n>>> Waiting 65s (rate-limit buffer) before return call...")
    await asyncio.sleep(65)

    await diag_call("RETURN", "BOM", "2026-11-22")


if __name__ == "__main__":
    asyncio.run(main())