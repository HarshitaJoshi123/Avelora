"""Try future_flights without date param — see if schema is different."""
import asyncio
import json
from mcp_client import aviation_mcp_call


async def test(args, label):
    print(f"\n{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print(f"Args: {json.dumps(args, indent=2)}\n")
    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            args,
        )
        payload = None
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    try:
                        payload = json.loads(item["text"])
                        break
                    except json.JSONDecodeError:
                        continue
        elif isinstance(result, dict):
            payload = result

        if payload is None:
            print(f"❌ Unparseable response")
            return
        print(f"ok: {payload.get('ok')}")
        if payload.get("ok") is not True:
            print(f"error: {payload.get('error')}")
        else:
            data = payload.get("data", [])
            print(f"data count: {len(data)}")
            if data:
                print(f"First record: {json.dumps(data[0], indent=2)[:300]}")
    except Exception as exc:
        print(f"❌ EXCEPTION: {type(exc).__name__}: {exc}")


async def main():
    # Test 1: minimal args (no date, no count)
    await test(
        {"airport_iata_code": "DEL", "schedule_type": "departure"},
        "Minimal — no date, no count",
    )

    # Test 2: with date but no count
    await test(
        {
            "airport_iata_code": "DEL",
            "schedule_type": "departure",
            "date": "2026-12-15",
        },
        "With date, no count",
    )

    # Test 3: with count but no date
    await test(
        {
            "airport_iata_code": "DEL",
            "schedule_type": "departure",
            "number_of_flights": 10,
        },
        "With count, no date",
    )


if __name__ == "__main__":
    asyncio.run(main())