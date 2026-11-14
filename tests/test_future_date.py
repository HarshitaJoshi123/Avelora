"""Prove/disprove: does future_flights endpoint work on free tier?"""
import asyncio
import json
from mcp_client import aviation_mcp_call


async def main():
    print("\n" + "=" * 70)
    print("TESTING: future_flights_arrival_departure_schedule")
    print("Calling with date=2026-12-15 (future)")
    print("=" * 70 + "\n")

    arguments = {
        "airport_iata_code": "DEL",
        "schedule_type": "departure",
        "date": "2026-12-15",   # Future date
        "number_of_flights": 100,
    }
    print(f"Arguments: {json.dumps(arguments, indent=2)}\n")

    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            arguments,
        )

        # Parse MCP content blocks
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

        print("=" * 70)
        print("RESULT")
        print("=" * 70)

        if payload is None:
            print("❌ Could not parse response")
            print("Raw:", result)
            return

        print(f"ok: {payload.get('ok')}")

        if payload.get("ok") is not True:
            error = str(payload.get("error", ""))
            print(f"error: {error}")

            print("\n" + "=" * 70)
            print("VERDICT")
            print("=" * 70)

            error_lower = error.lower()
            if "function_access_restricted" in error_lower:
                print("❌ CONFIRMED: future_flights endpoint is BLOCKED on free tier")
                print("   → My earlier claim was correct")
                print("   → Issue 2 fix (Google Flights) is mandatory")
            elif "rate_limit" in error_lower:
                print("⚠️  RATE LIMIT: quota exhausted for this endpoint")
                print("   → Try again after quota reset OR use Issue 2 fix")
            elif "subscription" in error_lower:
                print("❌ CONFIRMED: subscription plan doesn't support this endpoint")
                print("   → Issue 2 fix mandatory")
            else:
                print(f"❌ Unknown error: {error}")
            return

        # SUCCESS — got data!
        data = payload.get("data", [])
        print(f"data count: {len(data)}")

        print("\n" + "=" * 70)
        print("🎉 VERDICT: FUTURE DATES WORK!")
        print("=" * 70)
        print(f"Found {len(data)} flights for 2026-12-15")
        print("\nMy earlier claim was WRONG. Free tier gives future data.")
        print("You don't need Google Flights fallback for future dates.")

        if data:
            print("\nFirst 3 records:")
            for rec in data[:3]:
                dep_code = rec.get("departure_airport_code") or "DEL (assumed)"
                arr_code = rec.get("arrival_airport_code", "?")
                print(f"  • {rec.get('airline')} {rec.get('flight_number')}")
                print(f"    {dep_code} → {arr_code}")
                print(f"    {rec.get('departure_scheduled_time')} → "
                      f"{rec.get('arrival_scheduled_time')}")

    except Exception as exc:
        print(f"\n❌ EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())