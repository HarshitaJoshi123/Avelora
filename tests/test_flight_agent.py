import io
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

from backend import (
    budget_agent,
    extract_flight_fare_estimate,
    flight_agent,
    flight_matches_airline,
    flight_matches_departure_period,
    render_flight_section,
)


class FlightAgentTests(unittest.TestCase):
    def test_revision_departure_period_filters_schedule_times(self):
        self.assertTrue(
            flight_matches_departure_period(
                "2026-12-01T08:30:00+05:30",
                "morning",
            )
        )
        self.assertFalse(
            flight_matches_departure_period(
                "2026-12-01T19:30:00+05:30",
                "morning",
            )
        )
        self.assertFalse(
            flight_matches_departure_period("unknown", "morning")
        )

    def test_revision_airline_matches_schedule_and_search_results(self):
        self.assertTrue(
            flight_matches_airline(
                {"airline": "Air India", "flight_number": "AI101"},
                "Air India",
            )
        )
        self.assertFalse(
            flight_matches_airline(
                {"airline": "Example Air", "flight_number": "EA101"},
                "Air India",
            )
        )

    def run_agent(
        self,
        constraints,
        provider_responses,
        fare_results=None,
        user_query="",
        revision_flight_airline="",
        revision_agents=None,
    ):
        with (
            patch("backend.get_airport_code", side_effect=["DEL", "BOM"]),
            patch(
                "backend.aviation_mcp_call",
                new_callable=AsyncMock,
                side_effect=provider_responses,
            ) as aviation_call,
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                return_value=fare_results or {"results": []},
            ) as tavily_call,
            redirect_stdout(io.StringIO()),
        ):
            result = flight_agent({
                "trip_constraints": constraints,
                "user_query": user_query,
                "revision_flight_airline": revision_flight_airline,
                "revision_agents": revision_agents or [],
                "llm_calls": 0,
            })
        return result, aviation_call, tavily_call

    @staticmethod
    def flight_record(
        departure_code,
        arrival_code,
        departure_time,
        arrival_time,
        flight_number,
    ):
        record = {
            "arrival_airport_code": arrival_code,
            "departure_scheduled_time": departure_time,
            "arrival_scheduled_time": arrival_time,
            "airline": "Example Air",
            "flight_number": flight_number,
        }
        if departure_code is not None:
            record["departure_airport_code"] = departure_code
        return record

    def test_filters_wrong_date_and_unverified_or_mismatched_route_codes(self):
        requested_date = (date.today() + timedelta(days=10)).isoformat()
        wrong_date = (date.today() + timedelta(days=11)).isoformat()
        response = {
            "ok": True,
            "data": [
                self.flight_record(
                    "DEL", "BOM",
                    f"{requested_date}T08:00:00+05:30",
                    f"{requested_date}T10:00:00+05:30",
                    "EA101",
                ),
                self.flight_record(
                    "DEL", "BOM",
                    f"{wrong_date}T08:00:00+05:30",
                    f"{wrong_date}T10:00:00+05:30",
                    "EA102",
                ),
                self.flight_record(
                    None, "BOM",
                    f"{requested_date}T11:00:00+05:30",
                    f"{requested_date}T13:00:00+05:30",
                    "EA103",
                ),
                self.flight_record(
                    "", "BOM",
                    f"{requested_date}T12:00:00+05:30",
                    f"{requested_date}T14:00:00+05:30",
                    "EA104",
                ),
                self.flight_record(
                    "BLR", "BOM",
                    f"{requested_date}T15:00:00+05:30",
                    f"{requested_date}T17:00:00+05:30",
                    "EA105",
                ),
            ],
        }

        result, aviation_call, tavily_call = self.run_agent(
            {
                "origin": "Delhi",
                "destination": "Mumbai",
                "travel_date": requested_date,
            },
            [response],
        )

        self.assertEqual(len(result["flight_data"]), 1)
        self.assertEqual(result["flight_data"][0]["flight_number"], "EA101")
        self.assertIn(
            "Excluded 1 route-matching record(s) with a different departure date.",
            result["flight_results"],
        )
        self.assertNotIn("EA102", result["flight_results"])
        self.assertNotIn("EA103", result["flight_results"])
        self.assertNotIn("EA104", result["flight_results"])
        self.assertNotIn("EA105", result["flight_results"])
        aviation_call.assert_awaited_once()
        tavily_call.assert_awaited_once()

    def test_returns_matching_outbound_and_return_legs(self):
        outbound_date = (date.today() + timedelta(days=10)).isoformat()
        return_date = (date.today() + timedelta(days=14)).isoformat()
        outbound = {
            "ok": True,
            "data": [
                self.flight_record(
                    "DEL", "BOM",
                    f"{outbound_date}T08:00:00+05:30",
                    f"{outbound_date}T10:00:00+05:30",
                    "EA201",
                ),
            ],
        }
        returning = {
            "ok": True,
            "data": [
                self.flight_record(
                    "BOM", "DEL",
                    f"{return_date}T18:00:00+05:30",
                    f"{return_date}T20:00:00+05:30",
                    "EA202",
                ),
            ],
        }

        fare_results = {
            "results": [
                {
                    "title": "Delhi to Mumbai Flight Ticket Fare",
                    "url": "https://example.com/delhi-mumbai",
                    "content": (
                        f"Round-trip flight fare from Delhi to Mumbai "
                        f"on {outbound_date}, returning {return_date}: "
                        "INR 12,500 per adult."
                    ),
                    "score": 0.9,
                },
                {
                    "title": "Delhi to Mumbai flight booking price",
                    "url": "https://example.org/delhi-mumbai",
                    "content": (
                        f"Return flight ticket price Delhi to Mumbai on "
                        f"{outbound_date}, returning {return_date}: ₹13,000 per adult."
                    ),
                    "score": 0.8,
                },
            ],
        }
        result, aviation_call, tavily_call = self.run_agent(
            {
                "origin": "Delhi",
                "destination": "Mumbai",
                "travel_date": outbound_date,
                "return_date": return_date,
                "traveller_count": 2,
            },
            [outbound, returning],
            fare_results=fare_results,
            user_query="Plan a round trip for 1 person in economy.",
        )

        self.assertEqual(
            [(item["trip_leg"], item["flight_number"]) for item in result["flight_data"]],
            [("outbound", "EA201"), ("return", "EA202")],
        )
        self.assertEqual(aviation_call.await_count, 2)
        self.assertEqual(
            [call.args[1]["airport_iata_code"] for call in aviation_call.await_args_list],
            ["DEL", "BOM"],
        )
        self.assertEqual(
            [call.args[1]["date"] for call in aviation_call.await_args_list],
            [outbound_date, return_date],
        )
        self.assertEqual(
            result["flight_fare"]["status"],
            "indicative",
        )
        self.assertEqual(result["flight_fare"]["low_per_traveller_inr"], 12500)
        self.assertEqual(result["flight_fare"]["high_per_traveller_inr"], 13000)
        self.assertEqual(result["flight_fare"]["group_total_low_inr"], 25000)
        self.assertIn("for 2 traveller(s)", result["flight_results"])
        fare_query = tavily_call.await_args.args[0]
        self.assertIn("round-trip", fare_query)
        self.assertIn(outbound_date, fare_query)
        self.assertIn(return_date, fare_query)
        self.assertIn("2 adult traveller(s), economy class", fare_query)

        flight_section = render_flight_section(
            result["flight_data"],
            result["flight_results"],
            result["trip_constraints"],
            result["flight_fare"],
        )
        self.assertIn("Search-derived approximation only", flight_section)
        self.assertIn("https://example.com/delhi-mumbai", flight_section)

        with redirect_stdout(io.StringIO()):
            budget = budget_agent({
                "trip_constraints": {
                    "origin": "Delhi",
                    "destination": "Mumbai",
                    "travel_date": outbound_date,
                    "return_date": return_date,
                    "duration": "5 days",
                    "traveller_count": 2,
                },
                "user_query": "Plan a round trip for 1 person in economy.",
                "flight_fare": result["flight_fare"],
                "flight_data": result["flight_data"],
            })["budget_results"]
        self.assertIn("- Travellers: 2 traveller(s) specified", budget)
        self.assertIn("| Flights | 25,000 | 26,000 |", budget)
        self.assertIn("Estimated group total (including indicative airfare)", budget)
        tavily_call.assert_awaited_once()

    def test_revision_airline_filters_schedules_and_fare_sources(self):
        requested_date = (date.today() + timedelta(days=10)).isoformat()
        air_india_record = self.flight_record(
            "DEL", "BOM",
            f"{requested_date}T08:00:00+05:30",
            f"{requested_date}T10:00:00+05:30",
            "AI101",
        )
        air_india_record["airline"] = "Air India"
        response = {
            "ok": True,
            "data": [
                air_india_record,
                self.flight_record(
                    "DEL", "BOM",
                    f"{requested_date}T09:00:00+05:30",
                    f"{requested_date}T11:00:00+05:30",
                    "EA101",
                ),
            ],
        }
        fare_results = {
            "results": [
                {
                    "title": "Air India Delhi to Mumbai flight ticket fare",
                    "url": "https://example.com/air-india",
                    "content": (
                        f"One-way Air India flight fare from Delhi to Mumbai on "
                        f"{requested_date}: INR 7,500 per traveller."
                    ),
                },
                {
                    "title": "Example Air Delhi to Mumbai flight ticket fare",
                    "url": "https://example.com/example-air",
                    "content": (
                        f"One-way Example Air flight fare from Delhi to Mumbai on "
                        f"{requested_date}: INR 6,500 per traveller."
                    ),
                },
            ],
        }

        result, _, tavily_call = self.run_agent(
            {
                "origin": "Delhi",
                "destination": "Mumbai",
                "travel_date": requested_date,
            },
            [response],
            fare_results=fare_results,
            revision_flight_airline="Air India",
            revision_agents=["flight_agent"],
        )

        self.assertEqual(len(result["flight_data"]), 1)
        self.assertEqual(result["flight_data"][0]["airline"], "Air India")
        self.assertEqual(result["flight_fare"]["low_per_traveller_inr"], 7500)
        self.assertIn("Air India", tavily_call.await_args.args[0])

    def test_provider_error_is_reported_without_failing_the_call(self):
        outbound_date = (date.today() + timedelta(days=10)).isoformat()
        fare_results = {
            "results": [
                {
                    "title": "Delhi to Mumbai flight ticket fare",
                    "url": "https://example.com/delhi-mumbai-fare",
                    "content": (
                        f"One-way flight fare from Delhi to Mumbai on "
                        f"{outbound_date}: INR 5,432 per traveller."
                    ),
                },
            ],
        }

        result, aviation_call, tavily_call = self.run_agent(
            {
                "origin": "Delhi",
                "destination": "Mumbai",
                "travel_date": outbound_date,
            },
            [RuntimeError("AviationStack is unavailable")],
            fare_results=fare_results,
        )

        self.assertEqual(result["flight_data"], [])
        self.assertFalse(result["flight_results_available"])
        self.assertEqual(result["flight_fare"]["status"], "indicative")
        self.assertIn("OUTBOUND: Schedule unavailable", result["flight_results"])
        self.assertIn("AviationStack is unavailable", result["flight_results"])
        self.assertIn("Indicative one-way fare", result["flight_results"])
        aviation_call.assert_awaited_once()
        tavily_call.assert_awaited_once()

    def test_foreign_currency_fare_is_converted_before_budget_use(self):
        results = [
            {
                "title": "Delhi to Mumbai flight ticket fare",
                "url": "https://example.com/fare",
                "content": "Round-trip flight fare: C$500 per traveller.",
            },
        ]
        with patch(
            "backend.get_latest_inr_exchange_rate",
            return_value=(60.0, "2026-11-12"),
        ) as exchange_rate:
            fare = extract_flight_fare_estimate(
                results,
                "Delhi",
                "Mumbai",
                2,
            )

        self.assertEqual(fare["status"], "indicative")
        self.assertEqual(fare["low_per_traveller_inr"], 30000)
        self.assertEqual(fare["group_total_low_inr"], 60000)
        self.assertEqual(fare["sources"][0]["source_currency"], "CAD")
        exchange_rate.assert_called_once_with("CAD")

    def test_rejects_baggage_amount_even_with_route_date_and_passenger_unit(self):
        outbound_date = "2026-11-20"
        result = {
            "title": "Delhi to Mumbai flight ticket booking",
            "url": "https://example.com/fare",
            "content": (
                f"Baggage fee on {outbound_date}: INR 12,500 per traveller. "
                "Flight schedules are available."
            ),
        }

        fare = extract_flight_fare_estimate(
            [result],
            "Delhi",
            "Mumbai",
            1,
            outbound_date=outbound_date,
        )

        self.assertEqual(fare["status"], "unavailable")

    def test_rejects_fare_without_per_traveller_amount(self):
        outbound_date = "2026-11-20"
        result = {
            "title": "Delhi to Mumbai flight ticket fare",
            "url": "https://example.com/fare",
            "content": (
                f"Flight fare from Delhi to Mumbai on {outbound_date}: INR 12,500."
            ),
        }

        fare = extract_flight_fare_estimate(
            [result],
            "Delhi",
            "Mumbai",
            1,
            outbound_date=outbound_date,
        )

        self.assertEqual(fare["status"], "unavailable")

    def test_rejects_fare_that_does_not_match_requested_dates(self):
        result = {
            "title": "Delhi to Mumbai flight ticket fare",
            "url": "https://example.com/fare",
            "content": (
                "Flight fare from Delhi to Mumbai on 2026-11-21: "
                "INR 12,500 per adult."
            ),
        }

        fare = extract_flight_fare_estimate(
            [result],
            "Delhi",
            "Mumbai",
            1,
            outbound_date="2026-11-20",
        )

        self.assertEqual(fare["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
