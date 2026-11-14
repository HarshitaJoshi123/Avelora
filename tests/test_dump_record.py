"""Dump the actual record structure from AviationStack."""
import asyncio
import json
from mcp_client import aviation_mcp_call


async def main():
    print("\n" + "=" * 70)
    print("Dumping actual record structure from AviationStack")
    print("=" * 70 + "\n")

    arguments = {
        "airport_iata_code": "DEL",
        "schedule_type": "departure",
        "number_of_flights": 10,
    }

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
            print("❌ Could not parse response")
            return

        data = payload.get("data", [])
        if not data:
            print("No data returned")
            return

        # Dump FIRST record fully
        first = data[0]

        print("=" * 70)
        print("FIRST RECORD — FULL STRUCTURE")
        print("=" * 70)
        print(json.dumps(first, indent=2, default=str))

        print("\n" + "=" * 70)
        print("TOP-LEVEL KEYS IN FIRST RECORD")
        print("=" * 70)
        for key in sorted(first.keys()):
            value = first[key]
            value_type = type(value).__name__
            preview = str(value)[:80]
            print(f"  {key}: ({value_type}) {preview}")

        print("\n" + "=" * 70)
        print("NESTED KEYS (if any dict values)")
        print("=" * 70)
        for key, value in first.items():
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for subkey in sorted(value.keys()):
                    subvalue = value[subkey]
                    print(f"    {subkey}: ({type(subvalue).__name__}) {str(subvalue)[:60]}")

        # Try to find airport codes anywhere
        print("\n" + "=" * 70)
        print("SEARCHING FOR DEL→BOM IN SAMPLE")
        print("=" * 70)

        # Try various field names
        field_candidates = [
            ("departure_airport_code", "arrival_airport_code"),
            ("departure_iata", "arrival_iata"),
            ("dep_iata", "arr_iata"),
        ]

        for dep_field, arr_field in field_candidates:
            matches = [
                r for r in data
                if str(r.get(dep_field, "")).upper() == "DEL"
                and str(r.get(arr_field, "")).upper() == "BOM"
            ]
            if matches:
                print(f"✅ MATCH FOUND with ({dep_field}, {arr_field}): {len(matches)} flights")
                print("Sample:", json.dumps(matches[0], indent=2, default=str)[:500])
                return
            else:
                print(f"❌ No matches with ({dep_field}, {arr_field})")

        # Try nested departure/arrival dicts
        print("\n" + "=" * 70)
        print("CHECKING NESTED STRUCTURE (departure/arrival dicts)")
        print("=" * 70)
        for rec in data[:5]:
            dep = rec.get("departure", {})
            arr = rec.get("arrival", {})
            if isinstance(dep, dict) and isinstance(arr, dict):
                print(f"  departure.iata={dep.get('iata')}  arrival.iata={arr.get('iata')}")

    except Exception as exc:
        print(f"\n❌ EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())