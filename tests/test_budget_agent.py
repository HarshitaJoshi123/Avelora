import json
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from backend import _extract_researched_trip_costs, budget_agent


class BudgetAgentTests(unittest.TestCase):
    def run_budget(
        self,
        constraints,
        user_query="Estimate a trip budget for Jaipur.",
        hotel_data=None,
    ):
        state = {
            "trip_constraints": constraints,
            "user_query": user_query,
        }
        if hotel_data is not None:
            state["hotel_data"] = hotel_data
        with redirect_stdout(io.StringIO()):
            result = budget_agent(state)
        return result["budget_results"]

    def test_nights_only_duration_sets_nights_and_inclusive_days(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "duration": "5 nights",
        })
        self.assertIn("- Duration: 6 day(s), 5 night(s)", budget)
        self.assertIn("1 room(s) × 5 night(s)", budget)

    def test_conflicting_duration_uses_dates_and_reports_mismatch(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
            "duration": "5 days",
        })

        self.assertIn("- Duration: 3 day(s), 2 night(s)", budget)
        self.assertIn(
            "- Duration mismatch: requested 5 days, but the dates span "
            "3 day(s) and 2 night(s). Budget calculations use the dates.",
            budget,
        )
        self.assertIn("1 traveller(s) × 3 day(s) × ₹800–₹1,200/person/day", budget)
        self.assertIn("rough fallback, not a real-time price", budget)
        self.assertIn("1 room(s) × 2 night(s)", budget)

    def test_matching_duration_and_dates_do_not_report_mismatch(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
            "duration": "3 days",
        })

        self.assertIn("- Duration: 3 day(s), 2 night(s)", budget)
        self.assertNotIn("- Duration mismatch:", budget)

    def test_dates_without_duration_set_budget_length(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
        })

        self.assertIn("- Duration: 3 day(s), 2 night(s)", budget)
        self.assertIn("1 traveller(s) × 3 day(s)", budget)

    def test_structured_traveller_count_controls_budget(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "duration": "2 days",
            "traveller_count": 4,
        })

        self.assertIn("- Travellers: 4 traveller(s) specified", budget)
        self.assertIn("- Room assumption: 2 room(s)", budget)
        self.assertIn("4 traveller(s) × 2 day(s)", budget)

    def test_family_of_four_is_counted_without_reprompting(self):
        budget = self.run_budget(
            {"destination": "Jaipur", "duration": "2 days"},
            "Plan a 2-day trip to Jaipur for my family of four.",
        )

        self.assertIn("- Travellers: 4 traveller(s) specified", budget)
        self.assertIn("- Room assumption: 2 room(s)", budget)

    def test_missing_traveller_count_defaults_to_one_and_is_labeled(self):
        budget = self.run_budget({
            "destination": "Jaipur",
            "duration": "2 days",
        })

        self.assertIn(
            "- Travellers: 1 traveller (assumed; party size not specified)",
            budget,
        )

    def test_accommodation_uses_usable_matching_hotel_agent_rates(self):
        budget = self.run_budget(
            {
                "destination": "Jaipur",
                "duration": "3 days",
            },
            hotel_data=[
                {
                    "hotel_name": "Hotel Alpha",
                    "location": "Jaipur",
                    "price_per_night": 4000,
                    "currency": "INR",
                    "price_status": "indicative",
                    "price_source": "https://example.com/alpha",
                },
                {
                    "hotel_name": "Hotel Beta",
                    "location": "Jaipur",
                    "price_per_night": 6500,
                    "currency": "INR",
                    "price_status": "indicative_derived_from_total",
                    "total_stay_nights": 2,
                    "price_source": "https://example.com/beta",
                },
                {
                    "hotel_name": "Wrong stay length",
                    "location": "Jaipur",
                    "price_per_night": 9000,
                    "currency": "INR",
                    "price_status": "indicative_derived_from_total",
                    "total_stay_nights": 4,
                    "price_source": "https://example.com/wrong-stay",
                },
                {
                    "hotel_name": "Unavailable price",
                    "location": "Jaipur",
                    "price_per_night": 1000,
                    "currency": "INR",
                    "price_status": "unavailable",
                    "price_source": "https://example.com/unavailable",
                },
                {
                    "hotel_name": "Wrong destination",
                    "location": "Delhi",
                    "price_per_night": 2000,
                    "currency": "INR",
                    "price_status": "indicative",
                    "price_source": "https://example.com/delhi",
                },
            ],
        )

        self.assertIn("| Accommodation | 8,000 | 13,000 |", budget)
        self.assertIn("Range across 2 hotel-agent indicative rate(s)", budget)
        self.assertIn("Hotel Alpha, Hotel Beta", budget)
        self.assertIn("indicative: 4,000.00–4,000.00 INR per room/night", budget)
        self.assertIn("https://example.com/alpha", budget)
        self.assertIn("not date-specific quotes", budget)
        self.assertNotIn("Hotel rates are generic planning allowances", budget)

    def test_accommodation_uses_generic_fallback_without_usable_hotel_rates(self):
        budget = self.run_budget(
            {"destination": "Jaipur", "duration": "3 days"},
            hotel_data=[{
                "hotel_name": "Unpriced hotel",
                "location": "Jaipur",
                "price_per_night": None,
                "currency": "INR",
                "price_status": "unavailable",
                "price_source": "https://example.com/unpriced",
            }],
        )

        self.assertIn("| Accommodation | 5,000 | 10,000 |", budget)
        self.assertIn("Hotel rates are rough generic fallbacks", budget)

    def test_researched_cost_retains_source_currency_status_and_fx(self):
        source = {
            "title": "Local meal prices",
            "url": "https://example.com/food",
            "content": "Meals cost $10–$12 per person per day.",
        }
        extraction = {
            "estimates": [{
                "category": "food",
                "item": "meals",
                "low": 10,
                "high": 12,
                "currency": "USD",
                "unit": "per_person_per_day",
                "source_url": source["url"],
                "evidence": source["content"],
            }],
        }
        with (
            patch("backend._llm_text", return_value=json.dumps(extraction)),
            patch(
                "backend.get_latest_inr_exchange_rate",
                return_value=(80.0, "2026-11-12"),
            ),
        ):
            costs = _extract_researched_trip_costs(
                {"food": [source], "transport": [], "sightseeing": []},
                "## 5. Day-by-Day Itinerary",
            )

        estimate = costs["food"][0]
        self.assertEqual(estimate["source_low"], 10)
        self.assertEqual(estimate["source_high"], 12)
        self.assertEqual(estimate["currency"], "USD")
        self.assertEqual(estimate["low_inr"], 800)
        self.assertEqual(estimate["high_inr"], 960)
        self.assertEqual(estimate["exchange_rate_to_inr"], 80.0)
        self.assertEqual(estimate["exchange_rate_date"], "2026-11-12")
        self.assertEqual(estimate["status"], "indicative")
        self.assertEqual(estimate["source"]["url"], source["url"])
        self.assertEqual(estimate["source"]["evidence"], source["content"])

    def test_rejects_range_endpoints_from_different_source_prices(self):
        source = {
            "title": "Meal prices",
            "url": "https://example.com/meal-prices",
            "content": (
                "Meals cost $10-$20 per person per day. "
                "A separate service fee is $40."
            ),
        }
        extraction = {
            "estimates": [{
                "category": "food",
                "item": "meals",
                "low": 10,
                "high": 40,
                "currency": "USD",
                "unit": "per_person_per_day",
                "source_url": source["url"],
                "evidence": source["content"],
            }],
        }
        with patch("backend._llm_text", return_value=json.dumps(extraction)):
            costs = _extract_researched_trip_costs(
                {"food": [source], "transport": [], "sightseeing": []},
                "## 5. Day-by-Day Itinerary",
            )

        self.assertEqual(costs["food"], [])

    def test_rejects_third_party_attraction_ticket_source(self):
        source = {
            "title": "Official Amber Fort tickets",
            "url": "https://tickets.example.com/amber-fort",
            "content": "Amber Fort adult entry ticket: ₹100 per person per visit.",
        }
        extraction = {
            "estimates": [{
                "category": "sightseeing",
                "item": "Amber Fort",
                "low": 100,
                "high": 100,
                "currency": "INR",
                "unit": "per_person_per_visit",
                "source_url": source["url"],
                "evidence": source["content"],
            }],
        }
        with patch("backend._llm_text", return_value=json.dumps(extraction)):
            costs = _extract_researched_trip_costs(
                {"food": [], "transport": [], "sightseeing": [source]},
                """## 5. Day-by-Day Itinerary
| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Heritage | Amber Fort | Free time | Dinner |
""",
            )

        self.assertEqual(costs["sightseeing"], [])

    def test_attraction_price_source_must_name_the_planned_attraction(self):
        source = {
            "title": "Official Jaigarh Fort tickets",
            "url": "https://tourism.example.gov.in/jaigarh",
            "content": "Jaigarh Fort adult entry ticket: ₹100 per person per visit.",
        }
        extraction = {
            "estimates": [{
                "category": "sightseeing",
                "item": "Amber Fort",
                "low": 100,
                "high": 100,
                "currency": "INR",
                "unit": "per_person_per_visit",
                "source_url": source["url"],
                "evidence": source["content"],
            }],
        }
        with patch("backend._llm_text", return_value=json.dumps(extraction)):
            costs = _extract_researched_trip_costs(
                {"food": [], "transport": [], "sightseeing": [source]},
                """## 5. Day-by-Day Itinerary
| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Heritage | Amber Fort | Free time | Dinner |
""",
            )

        self.assertEqual(costs["sightseeing"], [])

    def test_researched_trip_costs_use_units_and_planned_attraction_visits(self):
        itinerary = """## 5. Day-by-Day Itinerary
| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Heritage | Amber Fort | Transfer from Amber Fort to City Palace | Local dinner |
"""
        search_results = {
            "food": (
                "Local meal prices",
                "https://example.com/food",
                "Local menu items cost ₹200–₹300 per person per meal. "
                "Typical food costs ₹700–₹800 per person per day.",
            ),
            "transport": (
                "Jaipur transport prices",
                "https://example.com/transport",
                "Amber Fort to City Palace costs ₹300–₹500 per person per ride.",
            ),
            "sightseeing": (
                "Official Amber Fort tickets",
                "https://tourism.example.gov.in/tickets",
                "Amber Fort adult entry ticket: ₹100 per person per visit.",
            ),
        }

        async def fake_search(query):
            lowered = query.lower()
            key = (
                "food" if "meal" in lowered
                else "transport" if "transport" in lowered
                else "sightseeing"
            )
            title, url, content = search_results[key]
            return [{
                "type": "text",
                "text": json.dumps({
                    "results": [{
                        "title": title,
                        "url": url,
                        "content": content,
                    }],
                }),
            }]

        extraction = {
            "estimates": [
                {
                    "category": "food",
                    "item": "local meals",
                    "low": 200,
                    "high": 300,
                    "currency": "INR",
                    "unit": "per_person_per_meal",
                    "source_url": "https://example.com/food",
                    "evidence": "Local menu items cost ₹200–₹300 per person per meal.",
                },
                {
                    "category": "food",
                    "item": "daily food cost",
                    "low": 700,
                    "high": 800,
                    "currency": "INR",
                    "unit": "per_person_per_day",
                    "source_url": "https://example.com/food",
                    "evidence": "Typical food costs ₹700–₹800 per person per day.",
                },
                {
                    "category": "transport",
                    "item": "Amber Fort to City Palace",
                    "low": 300,
                    "high": 500,
                    "currency": "INR",
                    "unit": "per_person_per_ride",
                    "source_url": "https://example.com/transport",
                    "evidence": search_results["transport"][2],
                },
                {
                    "category": "sightseeing",
                    "item": "Amber Fort",
                    "low": 100,
                    "high": 100,
                    "currency": "INR",
                    "unit": "per_person_per_visit",
                    "source_url": "https://tourism.example.gov.in/tickets",
                    "evidence": search_results["sightseeing"][2],
                },
            ],
        }
        with (
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=fake_search,
            ) as search,
            patch(
                "backend._llm_text",
                return_value=json.dumps(extraction),
            ) as extract,
            redirect_stdout(io.StringIO()),
        ):
            result = budget_agent({
                "trip_constraints": {
                    "destination": "Jaipur",
                    "duration": "1 day",
                    "traveller_count": 2,
                },
                "user_query": "Estimate a Jaipur trip budget.",
                "itinerary": itinerary,
            })

        budget = result["budget_results"]
        self.assertIn("| Food & dining | 1,200 | 1,800 |", budget)
        self.assertIn("| Local transportation | 600 | 1,000 |", budget)
        self.assertIn("| Sightseeing & entry tickets | 200 | 200 |", budget)
        self.assertIn("menu rates converted using 3 meals/day", budget)
        self.assertIn("₹300–₹500/person/ride", budget)
        self.assertIn("indicative: 200.00–300.00 INR per person/meal", budget)
        self.assertIn("Route-specific transport fares cover only the priced itinerary routes", budget)
        self.assertIn("Amber Fort (1 visit(s), ₹100–₹100/person/visit)", budget)
        self.assertIn("https://example.com/food", budget)
        self.assertIn("## 6. Estimated Budget", result["itinerary"])
        self.assertEqual(search.await_count, 3)
        extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
