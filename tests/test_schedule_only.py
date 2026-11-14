"""Direct test — only the schedule endpoint that works on free tier."""
import asyncio
import json
from mcp_client import aviation_mcp_call


async def main():
    print("\n" + "=" * 70)
    print("Testing flight_arrival_departure_schedule — DEL departures")
    print("This is the endpoint that flight_agent uses for TODAY'S date")
    print("=" * 70 + "\n")

    arguments = {
        "airport_iata_code": "DEL",
        "schedule_type": "departure",
        "number_of_flights": 100,
    }
    print(f"Arguments: {json.dumps(arguments, indent=2)}\n")

    try:
        result = await aviation_mcp_call(
            "flight_arrival_departure_schedule",
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

        if payload is None:
            print("❌ Could not parse MCP response")
            print("Raw:", result)
            return

        print("=" * 70)
        print("PARSED RESPONSE")
        print("=" * 70)
        print(f"ok: {payload.get('ok')}")
        print(f"context: {payload.get('context')}")

        if payload.get("ok") is not True:
            print(f"\n❌ ERROR: {payload.get('error')}")
            error = str(payload.get("error", "")).lower()
            if "function_access_restricted" in error:
                print("\nVERDICT: This endpoint is also blocked on your free tier.")
            elif "rate_limit" in error:
                print("\nVERDICT: Quota exhausted — wait until next month")
            return

        data = payload.get("data", [])
        print(f"data count: {len(data)}")

        if not data:
            print("\n" + "=" * 70)
            print("⚠️  VERDICT: API OK, but 0 flights returned")
            print("=" * 70)
            print("\nWhat this means:")
            print("  • Free tier gives access to this endpoint ✅")
            print("  • But the returned sample is EMPTY ❌")
            print("  • Likely reasons:")
            print("    - Free tier sample limit (only a few records shown)")
            print("    - Time of day matters — more flights later in the day")
            print("    - Sampling shows only a small subset of real flights")
            print("\nWhat to do:")
            print("  → Issue 2 fix (Google Flights link) is MANDATORY")
            print("  → Or upgrade to paid plan ($50/month)")
            return

        # Count DEL→BOM matches
        del_bom = [
            r for r in data
            if str(r.get("departure_airport_code", "")).upper() == "DEL"
            and str(r.get("arrival_airport_code", "")).upper() == "BOM"
        ]

        print("\n" + "=" * 70)
        print("FLIGHTS FOUND")
        print("=" * 70)
        print(f"Total records: {len(data)}")
        print(f"DEL → BOM matches: {len(del_bom)}")

        if del_bom:
            print("\n🎉 FLIGHT AGENT CAN WORK!")
            print("\nFirst 3 DEL→BOM flights:")
            for rec in del_bom[:3]:
                print(f"  • {rec.get('airline')} {rec.get('flight_number')}")
                print(f"    Departs: {rec.get('departure_scheduled_time')}")
                print(f"    Arrives: {rec.get('arrival_scheduled_time')}")
        else:
            routes = sorted(set(
                f"{r.get('departure_airport_code')}→{r.get('arrival_airport_code')}"
                for r in data[:50]
                if r.get('departure_airport_code') and r.get('arrival_airport_code')
            ))
            print("\n⚠️  No DEL→BOM in the returned sample")
            print(f"Routes present in sample (first 10):")
            for route in routes[:10]:
                print(f"  {route}")
            print("\nWhat this means:")
            print("  • Free tier sample doesn't include DEL→BOM flights")
            print("  • Each call returns a random/small subset")
            print("  • DEL→BOM exists in real life but not in this specific sample")

    except Exception as exc:
        print(f"\n❌ EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())