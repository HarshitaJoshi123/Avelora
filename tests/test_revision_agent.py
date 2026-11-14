import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from backend import plan_revision_research, revision_agent, revision_finalize_agent


class RevisionAgentTests(unittest.TestCase):
    def setUp(self):
        self.original_draft = """## 1. Trip Summary
Two days in Jaipur.
## 2. Flight Information
Not provided.
## 3. Hotel Suggestions
No verified hotel options.
## 4. Weather Information
No forecast available.
## 5. Day-by-Day Itinerary
| Day | Morning | Afternoon | Evening / Night |
|---|---|---|---|
| Day 1 - Arrival | Arrive | Visit the old city | Rest |
| Day 2 - Jaipur | Visit City Palace | Explore local markets | Dinner |
## 6. Estimated Budget
Budget estimate unavailable.
## 7. Final Recommendations
Confirm details before travel.
"""
        self.state = {
            "itinerary": self.original_draft,
            "human_feedback": "Replace City Palace with a park.",
            "user_query": "Plan a two-day trip to Jaipur.",
            "trip_constraints": {
                "destination": "Jaipur",
                "duration": "2 days",
            },
            "llm_calls": 3,
            "revision_count": 0,
        }

    @staticmethod
    def response(content):
        return SimpleNamespace(
            content=content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 25, "output_tokens": 40},
        )

    def test_hotel_feedback_updates_limit_and_schedules_research(self):
        plan = (
            '{"agents_to_rerun": ["hotel_agent"], '
            '"constraint_updates": {"hotel_budget": "INR 5000 per room/night"}, '
            '"flight_departure_preference": "", '
            '"flight_airline_preference": ""}'
        )
        with patch("backend._llm_text", return_value=plan):
            constraints, agents, flight_preference, airline_preference = plan_revision_research(
                "Replace the hotel with one under INR 5000 per night.",
                self.state["trip_constraints"],
                ["hotel_agent", "budget_agent", "itinerary_agent"],
            )

        self.assertEqual(
            constraints["hotel_budget"],
            "INR 5000 per room/night",
        )
        self.assertEqual(agents, ["hotel_agent", "budget_agent"])
        self.assertEqual(flight_preference, "")
        self.assertEqual(airline_preference, "")

    def test_itinerary_only_feedback_does_not_rerun_budget_agent(self):
        plan = (
            '{"agents_to_rerun": [], "constraint_updates": {}, '
            '"flight_departure_preference": "", '
            '"flight_airline_preference": ""}'
        )
        with patch("backend._llm_text", return_value=plan):
            _, agents, _, _ = plan_revision_research(
                "Make the itinerary wording more concise.",
                self.state["trip_constraints"],
                ["itinerary_agent", "budget_agent"],
            )

        self.assertEqual(agents, [])

    def test_destination_date_and_duration_changes_schedule_fresh_research(self):
        for field, value in (
            ("destination", "Paris"),
            ("travel_date", "2026-12-01"),
            ("duration", "3 days"),
        ):
            plan = (
                '{"agents_to_rerun": [], "constraint_updates": {'
                f'"{field}": "{value}"'
                '}, "flight_departure_preference": "", '
                '"flight_airline_preference": ""}'
            )
            with self.subTest(field=field), patch(
                "backend._llm_text",
                return_value=plan,
            ):
                _, agents, _, _ = plan_revision_research(
                    f"Change the trip {field}.",
                    self.state["trip_constraints"],
                    ["itinerary_agent"],
                )

            self.assertEqual(
                agents,
                ["flight_agent", "hotel_agent", "weather_agent"],
            )

    def test_flight_time_feedback_schedules_filtered_flight_research(self):
        plan = (
            '{"agents_to_rerun": ["flight_agent"], '
            '"constraint_updates": {}, '
            '"flight_departure_preference": "morning", '
            '"flight_airline_preference": ""}'
        )
        with patch("backend._llm_text", return_value=plan):
            _, agents, flight_preference, airline_preference = plan_revision_research(
                "Show me a morning flight.",
                self.state["trip_constraints"],
                ["flight_agent"],
            )

        self.assertEqual(agents, ["flight_agent"])
        self.assertEqual(flight_preference, "morning")
        self.assertEqual(airline_preference, "")

    def test_explicit_airline_feedback_schedules_flight_research(self):
        plan = (
            '{"agents_to_rerun": [], '
            '"constraint_updates": {}, '
            '"flight_departure_preference": "", '
            '"flight_airline_preference": "Air India"}'
        )
        with patch("backend._llm_text", return_value=plan):
            _, agents, flight_preference, airline_preference = plan_revision_research(
                "Please use Air India.",
                self.state["trip_constraints"],
                ["itinerary_agent"],
            )

        self.assertEqual(agents, ["flight_agent"])
        self.assertEqual(flight_preference, "")
        self.assertEqual(airline_preference, "Air India")

    def test_missing_day_section_retries_and_uses_revised_itinerary(self):
        incomplete = """## 1. Trip Summary
Two days in Jaipur.
## 2. Flight Information
Not provided.
## 3. Hotel Suggestions
No verified hotel options.
## 4. Weather Information
No forecast available.
## 6. Estimated Budget
Budget estimate unavailable.
## 7. Final Recommendations
Confirm details before travel.
"""
        revised = self.original_draft.replace(
            "Visit City Palace", "Visit a park"
        )

        with (
            patch("backend.llm") as mock_llm,
            patch(
                "backend._llm_text",
                return_value=(
                    '{"agents_to_rerun": [], "constraint_updates": {}, '
                    '"flight_departure_preference": "", '
                    '"flight_airline_preference": ""}'
                ),
            ),
            patch("backend.log_llm_usage"),
        ):
            mock_llm.invoke.side_effect = [
                self.response(incomplete),
                self.response(revised),
            ]
            with redirect_stdout(io.StringIO()) as output:
                result = revision_agent(self.state)

        invoke = mock_llm.invoke
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("Visit a park", result["itinerary"])
        self.assertNotIn("Visit City Palace", result["itinerary"])
        self.assertEqual(result["llm_calls"], 6)
        retry_prompt = invoke.call_args_list[1].args[0][1].content
        self.assertIn("Expected trip length: exactly 2 days", retry_prompt)
        self.assertIn("5. Day-by-Day Itinerary", retry_prompt)
        self.assertIn("| Day | Morning | Afternoon | Evening / Night |", retry_prompt)
        self.assertIn("Section 5 is missing.", retry_prompt)
        self.assertIn("preserving the user's requested revision", retry_prompt)
        debug_output = output.getvalue()
        self.assertIn("REVISION FIRST ATTEMPT", debug_output)
        self.assertIn("REVISION RETRY", debug_output)
        self.assertIn("input_tokens=25", debug_output)
        self.assertIn("output_tokens=40", debug_output)
        self.assertIn("complete generated response", debug_output)

    def test_missing_day_section_after_retry_fails_without_returning_old_draft(self):
        incomplete = """## 1. Trip Summary
Two days in Jaipur.
## 2. Flight Information
Not provided.
## 3. Hotel Suggestions
No verified hotel options.
## 4. Weather Information
No forecast available.
## 6. Estimated Budget
Budget estimate unavailable.
## 7. Final Recommendations
Confirm details before travel.
"""

        with (
            patch("backend.llm") as mock_llm,
            patch(
                "backend._llm_text",
                return_value=(
                    '{"agents_to_rerun": [], "constraint_updates": {}, '
                    '"flight_departure_preference": "", '
                    '"flight_airline_preference": ""}'
                ),
            ),
            patch("backend.log_llm_usage"),
        ):
            mock_llm.invoke.side_effect = [
                self.response(incomplete),
                self.response(incomplete),
            ]
            with redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"after one retry\. Validation failure: Section 5 is missing\.",
                ):
                    revision_agent(self.state)

        invoke = mock_llm.invoke
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("Visit City Palace", self.state["itinerary"])
        self.assertIn("complete generated response", output.getvalue())

    def test_revision_finalize_returns_revised_plan_as_final_response(self):
        final_plan = "Final revised itinerary"
        with patch(
            "backend.build_canonical_final_response",
            return_value=final_plan,
        ):
            result = revision_finalize_agent({
                "itinerary": "Revised draft",
                "trip_constraints": {"destination": "Jaipur"},
            })

        self.assertEqual(result["final_response"], final_plan)
        self.assertEqual(result["itinerary"], final_plan)
        self.assertEqual(result["messages"][0].content, final_plan)
        self.assertIn("final version", result["approval_request"])


if __name__ == "__main__":
    unittest.main()
