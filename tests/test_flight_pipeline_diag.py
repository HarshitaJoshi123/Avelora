# test_flight_pipeline_diag.py
import asyncio
import json
from datetime import date, datetime, timezone, timedelta

from mcp_client import aviation_mcp_call


IST = timezone(timedelta(hours=5, minutes=30))

# Same list as the planned cargo filter
CARGO_AIRLINES = {
    "blue dart", "blue dart aviation", "fedex", "ups", "dhl",
    "cathay cargo", "atlas air", "kalitta", "western global",
    "airbridgecargo", "aerologic", "cargolux", "lufthansa cargo",
    "emirates skycargo", "qatar airways cargo", "turkish cargo",
    "etihad cargo", "singapore airlines cargo", "korean air cargo",
    "asiana cargo", "japan airlines cargo", "cargojet",
}


def is_cargo(record):
    airline = str(record.get("airline") or "").lower()
    return any(c in airline for c in CARGO_AIRLINES)


def parse_scheduled_date(record):
    value = record.get("departure_scheduled_time")
    if not isinstance(value, str) or not value.strip():
        return None
    ts = value.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(IST).date()


def parse_payload(result):
    """Exactly what backend.parse_aviation_response does."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return None
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                try:
                    candidate = json.loads(item["text"])
                    if isinstance(candidate, dict):
                        return candidate
                except json.JSONDecodeError:
                    continue
    return None


async def diag_pipeline(leg_label, from_code, to_code, travel_date):
    print("\n" + "=" * 70)
    print(f"{leg_label}: {from_code} -> {to_code} on {travel_date}")
    print("=" * 70)

    arguments = {
        "airport_iata_code": from_code,
        "schedule_type": "departure",
        "date": travel_date,
        "number_of_flights": 100,
    }

    try:
        result = await aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            arguments,
        )
    except Exception as exc:
        print(f"CALL FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    print(f"Raw MCP return type: {type(result).__name__}")

    payload = parse_payload(result)
    if payload is None:
        print("Could not extract a dict payload.")
        print(f"Raw value preview: {str(result)[:500]}")
        return

    print(f"payload keys: {list(payload.keys())}")
    print(f"ok: {payload.get('ok')}")
    if payload.get("error"):
        print(f"error: {payload.get('error')}")

    records = payload.get("data", [])
    if not isinstance(records, list):
        print(f"data is not a list: {type(records).__name__}")
        return

    print(f"\nRAW RECORDS: {len(records)}")

    if not records:
        print("No records returned by API — nothing to filter.")
        return

    requested = date.fromisoformat(travel_date)
    to_c = to_code.upper()
    from_c = from_code.upper()

    counts = {
        "raw": len(records),
        "cargo_dropped": 0,
        "wrong_arrival": 0,
        "wrong_date": 0,
        "no_parsable_date": 0,
        "matches": 0,
    }
    matches = []

    for r in records:
        if is_cargo(r):
            counts["cargo_dropped"] += 1
            continue

        arr = str(r.get("arrival_airport_code") or "").upper()
        dep = str(r.get("departure_airport_code") or from_c).upper()

        if arr != to_c or dep != from_c:
            counts["wrong_arrival"] += 1
            continue

        sched = parse_scheduled_date(r)
        if sched is None:
            counts["no_parsable_date"] += 1
            continue
        if sched != requested:
            counts["wrong_date"] += 1
            continue

        counts["matches"] += 1
        matches.append(r)

    print("\nFILTER BREAKDOWN:")
    for k, v in counts.items():
        print(f"  {k:<20} : {v}")

    print("\nFIRST 10 RAW RECORDS:")
    for i, r in enumerate(records[:10]):
        print(
            f"  [{i}] airline={r.get('airline')!r} | "
            f"flight={r.get('flight_number')!r} | "
            f"dep={r.get('departure_airport_code')!r} | "
            f"arr={r.get('arrival_airport_code')!r} | "
            f"time={r.get('departure_scheduled_time')!r} | "
            f"ist_date={parse_scheduled_date(r)}"
        )

    print(f"\nMATCHED FLIGHTS ({len(matches)}):")
    for i, r in enumerate(matches[:10]):
        print(
            f"  [{i}] airline={r.get('airline')!r} | "
            f"flight={r.get('flight_number')!r} | "
            f"time={r.get('departure_scheduled_time')!r}"
        )


async def main():
    await diag_pipeline("OUTBOUND", "DEL", "BOM", "2026-11-20")
    print("\n>>> Waiting 65s (rate-limit buffer) before return call...")
    await asyncio.sleep(65)
    await diag_pipeline("RETURN", "BOM", "DEL", "2026-11-22")


if __name__ == "__main__":
    asyncio.run(main())