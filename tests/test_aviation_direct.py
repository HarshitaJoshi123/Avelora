import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

if not api_key:
    raise ValueError("AviationStack API key not found")

url = "https://api.aviationstack.com/v1/flights"

params = {
    "access_key": api_key,
    "dep_iata": "DEL",
    "limit": 5
}

try:
    print("Testing direct AviationStack API...")

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    print("HTTP Status:", response.status_code)

    data = response.json()

    if "error" in data:
        print("API ERROR:", data["error"])
    else:
        flights = data.get("data", [])
        print("Flights received:", len(flights))

        for flight in flights:
            print(
                flight.get("flight", {}).get("iata"),
                flight.get("airline", {}).get("name"),
                flight.get("departure", {}).get("iata"),
                "->",
                flight.get("arrival", {}).get("iata")
            )

except requests.exceptions.Timeout:
    print("Direct API request timed out.")

except Exception as e:
    print("ERROR:", repr(e))