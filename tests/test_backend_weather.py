from pprint import pprint
from backend import weather_agent

state = {
    "user_query": "Plan a trip to Mumbai",
    "trip_constraints": {
        "destination": "Mumbai"
    },
    "llm_calls": 0
}

print("\n========== TESTING WEATHER AGENT ==========\n")

result = weather_agent(state)

print("\n========== WEATHER RESULTS ==========\n")
print(result["weather_results"])

print("\n========== STRUCTURED WEATHER DATA ==========\n")
pprint(result["weather_data"])

print("\n========== LLM CALL COUNT ==========\n")
print(result["llm_calls"])