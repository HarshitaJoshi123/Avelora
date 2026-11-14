"""Test actual AviationStack rate limit window."""
import asyncio
import json
import time
from datetime import date, timedelta
from mcp_client import aviation_mcp_call


async def single_call(days_ahead):
    """One schedule call. Returns 'ok' or error string."""
    target = (date.today() + timedelta(days=days_ahead)).isoformat()
    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            {
                "airport_iata_code": "DEL",
                "schedule_type": "departure",
                "date": target,
                "number_of_flights": 5,
            },
        )
        payload = None
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    try:
                        payload = json.loads(item["text"])
                        break
                    except json.JSONDecodeError:
                        continue
        elif isinstance(result, dict):
            payload = result

        if payload is None:
            return "unparseable"
        if payload.get("ok") is True:
            return f"ok (data={len(payload.get('data', []))})"
        return str(payload.get("error", "unknown"))[:60]
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:50]}"


async def main():
    print("\n" + "=" * 70)
    print("Testing actual rate limit recovery window")
    print("=" * 70 + "\n")

    # Call 1 — baseline
    print("[T+0s]  Call 1:", await single_call(1))
    time.sleep(0.5)

    # Call 2 — immediate
    print("[T+0.5s] Call 2 (immediate):", await single_call(2))
    
    # Wait 3 seconds, try again
    print("\nWaiting 3s (Edit 1 delay)...")
    await asyncio.sleep(3)
    print("[T+3s]  Call 3 (after 3s):", await single_call(3))

    # Wait 15 more seconds, try again
    print("\nWaiting 15s (Edit 2 retry delay)...")
    await asyncio.sleep(15)
    print("[T+18s] Call 4 (after 18s total):", await single_call(4))

    # Wait 45 more seconds, try again
    print("\nWaiting 45s (testing 60s+ window)...")
    await asyncio.sleep(45)
    print("[T+63s] Call 5 (after 63s total):", await single_call(5))

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
Interpretation:
- Call 2 fails + Call 3 (3s) succeeds → Edit 1 alone fixes it
- Call 3 fails + Call 4 (18s) succeeds → Edit 2 needed (15s retry enough)
- Call 4 fails + Call 5 (63s) succeeds → Need 60s retry, my fix insufficient
- All fail → Quota/other issue, not rate limit
""")


if __name__ == "__main__":
    asyncio.run(main())