"""Verify the fix — should now find DEL→BOM flights."""
import asyncio
import json
from datetime import date
from mcp_client import aviation_mcp_call


async def main():
    print("\n" + "=" * 70)
    print("Testing with FIXED logic — should match DEL→BOM now")
    print("=" * 70 + "\n")

    # Same query as flight_agent
    arguments = {
        "airport_iata_code": "DEL",
        "schedule_type": "departure",
        "number_of_flights": 100,
    }

    result = await aviation_mcp_call(
        "flight_arrival_departure_schedule",
        arguments,
    )

    # Parse
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

    if payload is None or payload.get("ok") is not True:
        print("❌ API failed")
        return

    records = payload.get("data", [])
    print(f"Total records: {len(records)}\n")

    # THE FIX — mimic the patched logic
    leg_from = "DEL"
    leg_to = "BOM"
    requested_date = date.today()

    matches = []
    for record in records:
        arrival_code = str(record.get("arrival_airport_code", "")).upper()
        # THE FIX:
        departure_code = str(
            record.get("departure_airport_code") or leg_from
        ).strip().upper()

        # Also check date
        dep_time = record.get("departure_scheduled_time")
        if not dep_time:
            continue
        try:
            from datetime import datetime
            dep_date = datetime.fromisoformat(dep_time.replace("Z", "+00:00")).date()
        except ValueError:
            continue

        if arrival_code == leg_to and departure_code == leg_from and dep_date == requested_date:
            matches.append(record)

    print("=" * 70)
    print(f"DEL → BOM on {requested_date.isoformat()}: {len(matches)} flights")
    print("=" * 70)

    if matches:
        print("\n🎉 FIX WORKS! First 5 flights:\n")
        for rec in matches[:5]:
            print(f"  • {rec.get('airline')} {rec.get('flight_number')}")
            print(f"    Departs: {rec.get('departure_scheduled_time')}")
            print(f"    Arrives: {rec.get('arrival_scheduled_time')}")
            print(f"    Terminal: {rec.get('departure_terminal')} → {rec.get('arrival_terminal')}")
            print()
    else:
        print("\n⚠️  Still 0 matches — let me know, I'll investigate further")
        print("\nAll arrivals in sample:")
        all_arrivals = set(r.get("arrival_airport_code") for r in records if r.get("arrival_airport_code"))
        print(f"  {sorted(all_arrivals)[:20]}")


if __name__ == "__main__":
    asyncio.run(main())