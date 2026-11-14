"""Test multiple future dates to find boundary."""
import asyncio
import json
from datetime import date, timedelta
from mcp_client import aviation_mcp_call


async def test_date(target_date_str, label):
    arguments = {
        "airport_iata_code": "DEL",
        "schedule_type": "departure",
        "date": target_date_str,
        "number_of_flights": 5,
    }
    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            arguments,
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
            print(f"  {label:20s} ❌ Unparseable")
            return
        if payload.get("ok") is not True:
            err = str(payload.get("error", ""))[:60]
            print(f"  {label:20s} ❌ {err}")
        else:
            data = payload.get("data", [])
            print(f"  {label:20s} ✅ data={len(data)}")
    except Exception as exc:
        print(f"  {label:20s} 💥 {type(exc).__name__}: {str(exc)[:50]}")


async def main():
    today = date.today()
    print(f"\nToday: {today.isoformat()}")
    print("Testing future_flights with different offsets:\n")

    for days_ahead in [1, 3, 7, 14, 30, 60, 180]:
        target = (today + timedelta(days=days_ahead)).isoformat()
        await test_date(target, f"+{days_ahead}d ({target})")
        await asyncio.sleep(0.5)  # Rate limit politeness


if __name__ == "__main__":
    asyncio.run(main())