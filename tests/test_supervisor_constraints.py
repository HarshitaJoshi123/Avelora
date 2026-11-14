import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import (
    _itinerary_day_validation_error,
    deterministic_trip_constraints,
    expected_itinerary_day_count,
    extract_canonical_sections,
    format_day_by_day_itinerary,
    itinerary_agent,
    itinerary_has_all_requested_days,
    merge_deterministic_constraints,
    route_after_agent,
    route_after_revision,
    supervisor_agent,
)


class SupervisorConstraintTests(unittest.TestCase):
    def run_supervisor_with_selection(self, query, selected_agents):
        supervisor_response = json.dumps({
            "selected_agents": selected_agents,
            "trip_constraints": {
                "destination": "Jaipur",
                "duration": "5 days",
            },
            "reasoning": "Selected the requested travel specialists.",
        })
        with patch(
            "backend._llm_text",
            side_effect=[
                '{"allowed": true, "reason": ""}',
                supervisor_response,
            ],
        ):
            return supervisor_agent({
                "user_query": query,
                "llm_calls": 0,
            })

    def test_complete_trip_plan_adds_weather_without_dates(self):
        result = self.run_supervisor_with_selection(
            "Plan a 5-day trip to Jaipur with heritage sites, local food, "
            "comfortable stays and an estimated budget.",
            ["hotel_agent", "budget_agent", "itinerary_agent"],
        )

        self.assertIn("weather_agent", result["selected_agents"])
        self.assertEqual(result["trip_constraints"]["travel_date"], "")
        self.assertEqual(result["trip_constraints"]["return_date"], "")

    def test_hotel_only_request_does_not_automatically_add_weather(self):
        result = self.run_supervisor_with_selection(
            "Find comfortable stays in Jaipur.",
            ["hotel_agent", "itinerary_agent"],
        )

        self.assertNotIn("weather_agent", result["selected_agents"])

    def test_budget_runs_after_itinerary_and_before_approval(self):
        state = {
            "selected_agents": ["budget_agent", "itinerary_agent"],
        }

        self.assertEqual(route_after_agent("itinerary_agent")(state), "budget_agent")
        self.assertEqual(route_after_agent("budget_agent")(state), "human_approval")
        self.assertEqual(route_after_revision(state), "revision_finalize")

    def test_revision_routes_through_requested_agents_then_rebuilds(self):
        state = {
            "revision_agents": ["hotel_agent", "budget_agent"],
            "selected_agents": ["hotel_agent", "budget_agent"],
        }

        self.assertEqual(route_after_revision(state), "hotel_agent")
        self.assertEqual(route_after_agent("hotel_agent")(state), "budget_agent")
        self.assertEqual(
            route_after_agent("budget_agent")(state),
            "revision_finalize",
        )

    def test_revision_route_refreshes_all_destination_specialists(self):
        state = {
            "revision_agents": [
                "flight_agent",
                "hotel_agent",
                "weather_agent",
            ],
        }

        self.assertEqual(route_after_revision(state), "flight_agent")
        self.assertEqual(route_after_agent("flight_agent")(state), "hotel_agent")
        self.assertEqual(route_after_agent("hotel_agent")(state), "weather_agent")
        self.assertEqual(
            route_after_agent("weather_agent")(state),
            "revision_finalize",
        )

    def test_itinerary_day_count_uses_date_span_for_conflicts_and_missing_duration(self):
        mismatched_constraints = {
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
            "duration": "5 days",
        }
        date_only_constraints = {
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
            "duration": "",
        }
        three_day_table = """| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Arrival | Start | Visit | Rest |
| Day 2 - Exploring | Start | Visit | Rest |
| Day 3 - Departure | Start | Visit | Depart |
"""

        self.assertEqual(expected_itinerary_day_count(mismatched_constraints), 3)
        self.assertEqual(expected_itinerary_day_count(date_only_constraints), 3)
        self.assertTrue(
            itinerary_has_all_requested_days(three_day_table, mismatched_constraints)
        )
        self.assertFalse(
            itinerary_has_all_requested_days(
                three_day_table.replace("| Day 3 - Departure | Start | Visit | Depart |\n", ""),
                date_only_constraints,
            )
        )

    def test_day_table_accepts_evening_header_variants(self):
        constraints = {
            "travel_date": "2026-11-10",
            "return_date": "2026-11-12",
        }
        for evening_header in ("Evening/Night", "Evening", "Night"):
            table = f"""| Day | Morning | Afternoon | {evening_header} |
|---|---|---|---|
| Day 1 - Arrival | Start | Visit | Rest |
| Day 2 - Exploring | Start | Visit | Rest |
| Day 3 - Departure | Start | Visit | Depart |
"""
            with self.subTest(evening_header=evening_header):
                self.assertTrue(
                    itinerary_has_all_requested_days(table, constraints)
                )
                self.assertNotIn(
                    "could not be generated",
                    format_day_by_day_itinerary(table, expected_days=3),
                )

    def test_extract_canonical_sections_accepts_all_section_five_headings(self):
        headings = (
            "5. Day-by-Day Itinerary",
            "5) Day-by-Day Itinerary",
            "**5. Day-by-Day Itinerary**",
            "**5) Day-by-Day Itinerary**",
            "## 5. Day-by-Day Itinerary",
            "## **5. Day-by-Day Itinerary**",
            "Day-by-Day Itinerary",
            "**Day-by-Day Itinerary**",
        )
        for heading in headings:
            with self.subTest(heading=heading):
                sections = extract_canonical_sections(
                    f"{heading}\nThe itinerary body.\n"
                    "6. Estimated Budget\nThe budget body."
                )
                self.assertEqual(sections[5], "The itinerary body.")
                self.assertEqual(sections[6], "The budget body.")

        non_heading_content = (
            "| 5. Day-by-Day Itinerary | a table cell |\n"
            "This paragraph mentions 5. Day-by-Day Itinerary in passing."
        )
        self.assertNotIn(5, extract_canonical_sections(non_heading_content))

    def test_expected_itinerary_days_uses_duration_dates_or_unknown(self):
        self.assertEqual(
            expected_itinerary_day_count({
                "duration": "5 days and 4 nights",
            }),
            5,
        )
        self.assertEqual(
            expected_itinerary_day_count({
                "travel_date": "2026-11-10",
                "return_date": "2026-11-14",
            }),
            5,
        )
        self.assertIsNone(expected_itinerary_day_count({}))

    def test_day_validation_rejects_missing_duplicate_order_and_empty_cells(self):
        constraints = {"duration": "5 days and 4 nights"}

        def table(day_rows):
            return "\n".join([
                "| Day | Morning | Afternoon | Evening / Night |",
                "|---|---|---|---|",
                *day_rows,
            ])

        rows = [
            f"| Day {day} | Morning {day} | Afternoon {day} | Evening {day} |"
            for day in range(1, 6)
        ]
        self.assertTrue(itinerary_has_all_requested_days(table(rows), constraints))
        bold_rows = [
            f"| **Day {day}** | Morning {day} | Afternoon {day} | Evening {day} |"
            for day in range(1, 6)
        ]
        self.assertTrue(
            itinerary_has_all_requested_days(table(bold_rows), constraints)
        )

        cases = (
            (rows[:2] + rows[3:], "Missing Day 3."),
            (rows[:2] + [rows[1]] + rows[2:], "Duplicate Day 2."),
            (
                [rows[0], rows[2], rows[1], rows[3], rows[4]],
                "Incorrect day order:",
            ),
            (
                [rows[0], rows[1], "| Day 3 | | Afternoon 3 | Evening 3 |", *rows[3:]],
                "Missing morning activity for Day 3.",
            ),
        )
        for invalid_rows, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                error = _itinerary_day_validation_error(
                    table(invalid_rows),
                    constraints,
                )
                self.assertIsNotNone(error)
                self.assertIn(expected_error, error)

        self.assertIn(
            "Invalid table header",
            _itinerary_day_validation_error(
                table(rows).replace("Evening / Night", "Activities"),
                constraints,
            ),
        )
        self.assertIn(
            "Invalid separator row",
            _itinerary_day_validation_error(
                table(rows).replace("|---|---|---|---|", "|---|---|---|x|"),
                constraints,
            ),
        )
        self.assertIn(
            "Missing day number",
            _itinerary_day_validation_error(
                table(["| Arrival | Morning | Afternoon | Evening |", *rows[1:]]),
                constraints,
            ),
        )

    def test_non_table_daywise_format_remains_supported(self):
        daywise = """## **Day 1** - Arrival
Morning: Arrive
Afternoon: Check in
Evening: Dinner
## Day 2 - Departure
Morning: Breakfast
Afternoon: Depart
Evening: —
"""
        self.assertTrue(
            itinerary_has_all_requested_days(
                daywise,
                {"duration": "2 days"},
            )
        )

    def test_itinerary_retry_uses_user_budget_and_defers_detailed_estimate(self):
        complete_draft = """## 1. Trip Summary
Jaipur visit.
## 2. Flight Information
Not provided.
## 3. Hotel Suggestions
No verified options.
## 4. Weather Information
No forecast available.
## 5. Day-by-Day Itinerary
| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Arrival | Arrive | Check in | Rest |
| Day 2 - Heritage | Amber Fort | City Palace | Local dinner |
| Day 3 - Departure | Local breakfast | Flexible time | Depart |
## 6. Estimated Budget
The cost breakdown will be calculated from this itinerary.
## 7. Final Recommendations
Check prices before booking.
"""
        responses = [
            SimpleNamespace(
                content="Truncated draft",
                response_metadata={"finish_reason": "length"},
            ),
            SimpleNamespace(
                content=complete_draft,
                response_metadata={"finish_reason": "stop"},
            ),
        ]

        class RetryLLM:
            def __init__(self):
                self.prompts = []

            def invoke(self, messages):
                self.prompts.append(messages[-1].content)
                return responses.pop(0)

            def bind(self, **kwargs):
                return self

        llm = RetryLLM()
        canonical_call = {}

        def capture_canonical(**kwargs):
            canonical_call.update(kwargs)
            return kwargs["draft"]

        with (
            patch("backend.llm", llm),
            patch("backend.log_llm_usage"),
            patch(
                "backend.build_canonical_final_response",
                side_effect=capture_canonical,
            ),
        ):
            result = itinerary_agent({
                "user_query": (
                    "Plan a 5-day Jaipur trip from 2026-11-10 through "
                    "2026-11-12 under ₹20,000."
                ),
                "trip_constraints": {
                    "destination": "Jaipur",
                    "travel_date": "2026-11-10",
                    "return_date": "2026-11-12",
                    "duration": "5 days",
                    "hotel_budget": "₹5,000 per night",
                },
                "selected_agents": ["itinerary_agent", "budget_agent"],
                "budget_results": "STALE BUDGET THAT MUST NOT GUIDE THE PLAN",
            })

        self.assertEqual(len(llm.prompts), 2)
        for prompt in llm.prompts:
            self.assertIn("₹20,000", prompt)
            self.assertIn("detailed budget", prompt.lower())
            self.assertIn("duration': '3 days'", prompt)
            self.assertIn("date span as authoritative", prompt)
            self.assertNotIn("STALE BUDGET", prompt)
        self.assertEqual(canonical_call["budget_section"], "")
        self.assertIn("Day 3 - Departure", result["itinerary"])

    def test_clear_user_values_override_supervisor_extraction(self):
        merged = merge_deterministic_constraints(
            {
                "origin": "Paris",
                "destination": "Rome",
                "travel_date": "2026-11-20",
                "return_date": "2026-11-25",
                "duration": "2 days",
                "travel_style": "luxury",
            },
            "Plan a 4-day trip from Delhi to Goa, 10-14 November 2026, "
            "comfortable travel.",
        )

        self.assertEqual(merged["origin"], "Delhi")
        self.assertEqual(merged["destination"], "Goa")
        self.assertEqual(merged["travel_date"], "2026-11-10")
        self.assertEqual(merged["return_date"], "2026-11-14")
        self.assertEqual(merged["duration"], "4 days")
        self.assertEqual(merged["travel_style"], "comfortable")

    def test_traveller_count_is_extracted_from_numeric_and_family_phrasing(self):
        numeric = deterministic_trip_constraints(
            "Plan a Jaipur trip for 2 people."
        )
        family = deterministic_trip_constraints(
            "Plan a Jaipur trip for my family of four."
        )
        unspecified = deterministic_trip_constraints(
            "Plan a Jaipur trip."
        )

        self.assertEqual(numeric["traveller_count"], 2)
        self.assertEqual(family["traveller_count"], 4)
        self.assertIsNone(unspecified["traveller_count"])

    def test_explicit_party_size_preserves_adult_child_total(self):
        merged = merge_deterministic_constraints(
            {"traveller_count": 3},
            "Plan a trip for 2 adults and 1 child.",
        )

        self.assertEqual(merged["traveller_count"], 3)

    def test_ambiguous_values_do_not_override_supervisor_extraction(self):
        extracted = {
            "origin": "Jaipur",
            "destination": "Kochi",
            "travel_date": "2026-11-20",
            "return_date": "2026-11-25",
            "duration": "6 days",
            "travel_style": "mid-range",
        }

        merged = merge_deterministic_constraints(
            extracted,
            "Consider a trip from Delhi to Goa and Mumbai; "
            "an event is 2026-11-10 and travel may be 2026-11-14; "
            "it could be 3 days or 5 nights, luxury or budget-friendly.",
        )

        for field, value in extracted.items():
            with self.subTest(field=field):
                self.assertEqual(merged[field], value)

    def test_multiple_destinations_are_not_treated_as_one_route(self):
        constraints = deterministic_trip_constraints(
            "Plan a trip from Delhi to Goa, then Mumbai."
        )

        self.assertEqual(constraints["origin"], "")
        self.assertEqual(constraints["destination"], "")

    def test_unrelated_multiple_dates_are_not_treated_as_trip_dates(self):
        constraints = deterministic_trip_constraints(
            "My event is 2026-11-10, and the trip might be 2026-11-14."
        )

        self.assertEqual(constraints["travel_date"], "")
        self.assertEqual(constraints["return_date"], "")

    def test_explicit_return_date_pair_is_extracted(self):
        constraints = deterministic_trip_constraints(
            "Travel from Delhi to Goa on 2026-11-10, returning on 2026-11-14."
        )

        self.assertEqual(constraints["travel_date"], "2026-11-10")
        self.assertEqual(constraints["return_date"], "2026-11-14")


if __name__ == "__main__":
    unittest.main()
