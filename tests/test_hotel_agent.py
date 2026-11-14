import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from backend import (
    classify_hotel_url,
    extract_hotel_prices,
    hotel_budget_match_status,
    hotel_agent,
    is_real_hotel_candidate,
    render_hotel_section,
    sanitize_verified_hotel_data,
    validate_hotel_price_for_display,
    verify_discovered_hotel_names,
)


class HotelAgentSearchStatusTests(unittest.TestCase):
    search_queries = [
        ("Mid-range", "mid-range properties in Mumbai"),
        ("Heritage", "heritage properties in Mumbai"),
    ]

    def run_agent(self, search_responses):
        with (
            patch("backend.hotel_category_queries", return_value=self.search_queries),
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=search_responses,
            ) as tavily_search,
            patch("backend.is_real_hotel_candidate", return_value=True),
            patch("backend.classify_hotel_url", return_value="official_property_site"),
            patch("backend.discover_hotel_names_from_search_results", return_value=[]),
            redirect_stdout(io.StringIO()),
        ):
            result = hotel_agent({
                "trip_constraints": {"destination": "Mumbai"},
                "llm_calls": 0,
            })
        return result, tavily_search

    def test_all_category_searches_fail_reports_unavailable(self):
        result, tavily_search = self.run_agent([
            RuntimeError("Tavily connection timed out"),
            RuntimeError("Tavily service unavailable"),
        ])

        self.assertEqual(result["hotel_data"], [])
        self.assertIn(
            "Hotel search is temporarily unavailable; no results could be checked.",
            result["hotel_results"],
        )
        self.assertNotIn("No hotel properties could be verified", result["hotel_results"])
        self.assertEqual(tavily_search.await_count, 2)

    def test_successful_searches_with_no_candidates_reports_no_verified_properties(self):
        result, tavily_search = self.run_agent([
            {"results": []},
            {"results": []},
        ])

        self.assertEqual(result["hotel_data"], [])
        self.assertIn(
            "No hotel properties could be verified for Mumbai in this search.",
            result["hotel_results"],
        )
        self.assertNotIn("temporarily unavailable", result["hotel_results"])
        self.assertEqual(tavily_search.await_count, 2)

    def test_partial_category_failure_keeps_found_property_and_discloses_partial_search(self):
        property_result = {
            "results": [
                {
                    "title": "Example Residency Mumbai",
                    "url": "https://example-residency.com/mumbai",
                    "content": "Hotel in Mumbai with guest rooms and property address.",
                    "score": 0.9,
                }
            ]
        }
        result, tavily_search = self.run_agent([
            property_result,
            RuntimeError("Tavily temporary failure"),
            {"results": []},
        ])

        self.assertEqual(len(result["hotel_data"]), 1)
        self.assertEqual(result["hotel_data"][0]["hotel_name"], "Example Residency Mumbai")
        self.assertIn("Hotel search was partial: 1 of 2 category searches failed.", result["hotel_results"])
        self.assertIn("The properties shown came from successful searches.", result["hotel_results"])
        self.assertEqual(tavily_search.await_count, 3)

    def test_shortlist_fills_to_five_when_verified_candidates_share_one_category(self):
        property_results = {
            "results": [
                {
                    "title": f"Verified Hotel {index} Mumbai",
                    "url": f"https://example.com/hotel-{index}",
                    "content": "Hotel property in Mumbai with guest rooms.",
                    "score": 0.9,
                }
                for index in range(6)
            ]
        }
        empty_results = {"results": []}

        result, tavily_search = self.run_agent(
            [property_results, empty_results]
            + [empty_results for _ in range(5)]
        )

        self.assertEqual(len(result["hotel_data"]), 5)
        self.assertEqual(
            [hotel["hotel_name"] for hotel in result["hotel_data"]],
            [f"Verified Hotel {index} Mumbai" for index in range(5)],
        )
        self.assertEqual(tavily_search.await_count, 7)


class HotelCandidateVerificationTests(unittest.TestCase):
    def test_best_areas_and_places_to_visit_article_is_not_a_hotel(self):
        result = {
            "title": "Best Areas to Stay in Jaipur & Places to Visit | Rudra Vilas",
            "url": (
                "https://rudravilasjaipur.in/"
                "best-areas-to-stay-in-jaipur-for-first-time-visitors-and-places-to-visit"
            ),
            "content": (
                "Rudra Vilas is a heritage hotel located in Jaipur, "
                "with guest rooms and a property address."
            ),
            "score": 0.9,
        }

        self.assertFalse(is_real_hotel_candidate(result, "Jaipur"))
        self.assertEqual(
            classify_hotel_url(result["url"], "Rudra Vilas"),
            "unverified",
        )

    def test_actual_rudra_vilas_property_page_remains_eligible(self):
        result = {
            "title": "Rudra Vilas",
            "url": "https://rudravilasjaipur.in/rooms/",
            "content": "Rudra Vilas is a hotel located in Jaipur with guest rooms.",
            "score": 0.9,
        }

        self.assertTrue(is_real_hotel_candidate(result, "Jaipur"))
        self.assertEqual(
            classify_hotel_url(result["url"], "Rudra Vilas"),
            "official_property_site",
        )

    def test_sparse_property_page_needs_independent_destination_evidence(self):
        property_page = {
            "results": [
                {
                    "title": "The Courtyard Retreat",
                    "url": "https://courtyardretreat.example/rooms",
                    "content": "Elegant rooms and dining.",
                    "score": 0.9,
                }
            ]
        }
        location_confirmation = {
            "results": [
                {
                    "title": "The Courtyard Retreat in Jaipur",
                    "url": "https://travel.example.com/courtyard-retreat-jaipur",
                    "content": "The Courtyard Retreat is located in Jaipur.",
                    "score": 0.8,
                }
            ]
        }

        with (
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[property_page, location_confirmation],
            ) as tavily_search,
            redirect_stdout(io.StringIO()),
        ):
            verified = verify_discovered_hotel_names(
                [{"name": "The Courtyard Retreat", "category": "Heritage"}],
                "Jaipur",
                [],
            )

        self.assertEqual(len(verified), 1)
        self.assertEqual(
            verified[0]["url"],
            "https://courtyardretreat.example/rooms",
        )
        self.assertIn("located in Jaipur", verified[0]["content"])
        self.assertEqual(tavily_search.await_count, 2)

        sanitized = sanitize_verified_hotel_data(
            [{
                "hotel_name": verified[0]["_hotel_name"],
                "booking_url": verified[0]["url"],
                "search_snippet": verified[0]["content"],
                "property_page_verified": verified[0]["property_page_verified"],
            }],
            "Jaipur",
        )
        self.assertEqual(len(sanitized), 1)

    def test_sparse_property_page_without_city_confirmation_is_rejected(self):
        property_page = {
            "results": [
                {
                    "title": "The Courtyard Retreat",
                    "url": "https://courtyardretreat.example/rooms",
                    "content": "Elegant rooms and dining.",
                    "score": 0.9,
                }
            ]
        }
        unrelated_location = {
            "results": [
                {
                    "title": "The Courtyard Retreat in Udaipur",
                    "url": "https://travel.example.com/courtyard-retreat-udaipur",
                    "content": "The Courtyard Retreat is located in Udaipur.",
                    "score": 0.8,
                }
            ]
        }
        empty_results = {"results": []}

        with (
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[property_page, unrelated_location, empty_results],
            ),
            redirect_stdout(io.StringIO()),
        ):
            verified = verify_discovered_hotel_names(
                [{"name": "The Courtyard Retreat", "category": "Heritage"}],
                "Jaipur",
                [],
            )

        self.assertEqual(verified, [])

    def test_generic_review_page_is_not_accepted_as_property_page(self):
        review_page = {
            "results": [
                {
                    "title": "The Courtyard Retreat Review",
                    "url": "https://courtyardretreat.example/reviews",
                    "content": "A travel review of rooms and dining.",
                    "score": 0.9,
                }
            ]
        }
        location_confirmation = {
            "results": [
                {
                    "title": "The Courtyard Retreat in Jaipur",
                    "url": "https://travel.example.com/courtyard-retreat-jaipur",
                    "content": "The Courtyard Retreat is located in Jaipur.",
                    "score": 0.8,
                }
            ]
        }

        with (
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[review_page, location_confirmation, {"results": []}],
            ),
            redirect_stdout(io.StringIO()),
        ):
            verified = verify_discovered_hotel_names(
                [{"name": "The Courtyard Retreat", "category": "Heritage"}],
                "Jaipur",
                [],
            )

        self.assertEqual(verified, [])


class HotelPriceCurrencyTests(unittest.TestCase):
    def test_extracts_supported_foreign_currency_nightly_price(self):
        extracted = extract_hotel_prices("USD 100 per night")

        self.assertEqual(extracted["currency"], "USD")
        self.assertEqual(extracted["price_per_night"], 100)

    def test_extracts_total_stay_nights_and_unqualified_price(self):
        total = extract_hotel_prices("Total price USD 400 for 4 nights")
        unqualified = extract_hotel_prices("Rooms from USD 100")

        self.assertEqual(total["total_stay_price"], 400)
        self.assertEqual(total["total_stay_nights"], 4)
        self.assertEqual(unqualified["price_indication"], 100)

    def test_foreign_price_is_converted_using_reference_rate(self):
        with patch(
            "backend.get_latest_inr_exchange_rate",
            return_value=(83.5, "2026-11-11"),
        ):
            converted = validate_hotel_price_for_display(100, "USD")

        self.assertEqual(
            converted,
            (8350.0, 83.5, "2026-11-11", "indicative"),
        )

    def test_budget_is_compared_in_inr_even_when_source_price_is_foreign(self):
        with patch(
            "backend.get_latest_inr_exchange_rate",
            return_value=(83.5, "2026-11-11"),
        ) as exchange_rate:
            status = hotel_budget_match_status(8350, "USD 90-110")

        self.assertEqual(status, "within_budget")
        exchange_rate.assert_called_once_with("USD")

    def test_inr_budget_compares_directly_with_converted_foreign_hotel_price(self):
        with patch("backend.get_latest_inr_exchange_rate") as exchange_rate:
            status = hotel_budget_match_status(8350, "INR 8000-9000")

        self.assertEqual(status, "within_budget")
        exchange_rate.assert_not_called()

    def test_budget_conversion_failure_is_reported_as_conversion_unavailable(self):
        with patch(
            "backend.get_latest_inr_exchange_rate",
            side_effect=RuntimeError("rate service unavailable"),
        ):
            status = hotel_budget_match_status(8350, "USD 90-110")

        self.assertEqual(status, "conversion_unavailable")

    def test_hotel_agent_displays_inr_and_original_currency_without_date_claim(self):
        property_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Hotel rooms in Jaipur.",
                "score": 0.9,
            }]
        }
        price_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Rooms from USD 100 per night.",
                "score": 0.9,
            }]
        }

        with (
            patch(
                "backend.hotel_category_queries",
                return_value=[("Mid-range", "find a named hotel")],
            ),
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[property_result, price_result],
            ) as tavily_search,
            patch("backend.is_real_hotel_candidate", return_value=True),
            patch("backend.classify_hotel_url", return_value="official_property_site"),
            patch("backend.discover_hotel_names_from_search_results", return_value=[]),
            patch(
                "backend.get_latest_inr_exchange_rate",
                return_value=(83.5, "2026-11-11"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = hotel_agent({
                "trip_constraints": {
                    "destination": "Jaipur",
                    "travel_date": "2026-12-10",
                    "return_date": "2026-12-12",
                    "hotel_budget": "USD 90-110",
                },
                "llm_calls": 0,
            })

        hotel = result["hotel_data"][0]
        self.assertEqual(hotel["price_per_night"], 8350.0)
        self.assertEqual(hotel["currency"], "INR")
        self.assertEqual(hotel["original_price_per_night"], 100)
        self.assertEqual(hotel["original_currency"], "USD")
        self.assertEqual(hotel["exchange_rate_date"], "2026-11-11")
        self.assertEqual(hotel["budget_match_status"], "within_budget")
        self.assertIn("INR 8,350.00 per night (USD 100.00)", result["hotel_results"])
        self.assertNotIn("not date-specific", result["hotel_results"])
        self.assertNotIn("2026-12-10", tavily_search.await_args_list[1].args[0])

        rendered = render_hotel_section(result["hotel_data"])
        self.assertIn(
            "| Hotel | Category | Indicative Price | Remarks | Booking / Property Link |",
            rendered,
        )
        self.assertNotIn("| Currency |", rendered)
        self.assertIn("INR 8,350.00 per night (USD 100.00)", rendered)
        self.assertNotIn("latest search-derived indication", rendered)
        self.assertIn("reference rate dated 2026-11-11", rendered)
        self.assertIn(
            "Prices are estimates, not live quotes. Please confirm dates and availability before booking.",
            rendered,
        )

    def test_total_stay_price_is_used_to_estimate_per_night_price(self):
        property_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Hotel rooms in Jaipur.",
                "score": 0.9,
            }]
        }
        price_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Total price USD 400 for 4 nights.",
                "score": 0.9,
            }]
        }

        with (
            patch(
                "backend.hotel_category_queries",
                return_value=[("Mid-range", "find a named hotel")],
            ),
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[property_result, price_result],
            ),
            patch("backend.is_real_hotel_candidate", return_value=True),
            patch("backend.classify_hotel_url", return_value="official_property_site"),
            patch("backend.discover_hotel_names_from_search_results", return_value=[]),
            patch(
                "backend.get_latest_inr_exchange_rate",
                return_value=(83.5, "2026-11-11"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = hotel_agent({
                "trip_constraints": {"destination": "Jaipur"},
                "llm_calls": 0,
            })

        hotel = result["hotel_data"][0]
        self.assertEqual(hotel["price_per_night"], 8350)
        self.assertEqual(hotel["total_stay_price"], 33400)
        self.assertEqual(hotel["total_stay_nights"], 4)
        self.assertEqual(hotel["price_status"], "indicative_derived_from_total")
        self.assertIn("INR 8,350.00 per night", result["hotel_results"])
        self.assertIn("USD 400.00 total for 4 nights", result["hotel_results"])

        rendered = render_hotel_section(result["hotel_data"])
        self.assertIn("Nightly estimate based on a total for 4 nights", rendered)
        self.assertIn("USD 400.00 total for 4 nights", rendered)

    def test_unqualified_listed_price_is_shown_without_treating_it_as_nightly(self):
        property_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Hotel rooms in Jaipur.",
                "score": 0.9,
            }]
        }
        price_result = {
            "results": [{
                "title": "Courtyard Retreat Jaipur Hotel",
                "url": "https://courtyardretreat.example/jaipur-hotel",
                "content": "Rooms from USD 100.",
                "score": 0.9,
            }]
        }

        with (
            patch(
                "backend.hotel_category_queries",
                return_value=[("Mid-range", "find a named hotel")],
            ),
            patch(
                "backend.tavily_mcp_search",
                new_callable=AsyncMock,
                side_effect=[property_result, price_result],
            ),
            patch("backend.is_real_hotel_candidate", return_value=True),
            patch("backend.classify_hotel_url", return_value="official_property_site"),
            patch("backend.discover_hotel_names_from_search_results", return_value=[]),
            patch(
                "backend.get_latest_inr_exchange_rate",
                return_value=(83.5, "2026-11-11"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = hotel_agent({
                "trip_constraints": {
                    "destination": "Jaipur",
                    "hotel_budget": "USD 90-110",
                },
                "llm_calls": 0,
            })

        hotel = result["hotel_data"][0]
        self.assertIsNone(hotel["price_per_night"])
        self.assertEqual(hotel["listed_price_inr"], 8350)
        self.assertEqual(hotel["listed_price_unit"], "price period not stated")
        self.assertEqual(hotel["budget_match_status"], "not_verified")
        self.assertIn("INR 8,350.00 (USD 100.00); price period not stated", result["hotel_results"])
        rendered = render_hotel_section(result["hotel_data"])
        self.assertIn("INR 8,350.00 (USD 100.00); price period not stated", rendered)
        self.assertNotIn("not compared with a nightly budget", rendered)


if __name__ == "__main__":
    unittest.main()
