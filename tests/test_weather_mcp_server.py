import unittest
from unittest.mock import AsyncMock, patch

from custom_weather_mcp_server import get_current_weather, get_forecast
from backend import (
    fallback_agent_selection,
    _resolve_weather_forecast_dates,
    render_weather_section,
    weather_agent,
)


class WeatherForecastWindowTests(unittest.TestCase):
    def test_weather_without_trip_dates_is_labeled_as_upcoming_outlook(self):
        rendered = render_weather_section(
            {
                "city": "Jaipur",
                "current_weather": None,
                "forecast": [{
                    "datetime": "2026-11-12 12:00:00",
                    "temperature_c": 24,
                    "condition": "clear",
                }],
            },
            {"destination": "Jaipur", "travel_date": "", "return_date": ""},
        )

        self.assertIn(
            "**Upcoming forecast (Nov 12):**",
            rendered,
        )
        self.assertIn(
            "**Note:** No travel dates were provided, so this is not a forecast "
            "for your trip.",
            rendered,
        )
        self.assertNotIn("Forecast entries for requested dates", rendered)
        self.assertNotIn("2026-11-12 12:00:00", rendered)

    def test_weather_summary_is_compact_and_highlights_rain(self):
        rendered = render_weather_section(
            {
                "city": "Jaipur",
                "current_weather": {
                    "temperature_c": 21.62,
                    "condition": "haze",
                },
                "forecast": [
                    {
                        "datetime": "2026-11-12 18:00:00",
                        "destination_local_datetime": "2026-11-12 23:30:00+05:30",
                        "temperature_c": 21.62,
                        "condition": "broken clouds",
                    },
                    {
                        "datetime": "2026-11-13 08:00:00",
                        "destination_local_datetime": "2026-11-13 13:30:00+05:30",
                        "temperature_c": 30.86,
                        "condition": "clear sky",
                    },
                    {
                        "datetime": "2026-11-15 12:00:00",
                        "destination_local_datetime": "2026-11-15 17:30:00+05:30",
                        "temperature_c": 24.01,
                        "condition": "light rain",
                    },
                    {
                        "datetime": "2026-11-15 15:00:00",
                        "destination_local_datetime": "2026-11-15 20:30:00+05:30",
                        "temperature_c": 23.22,
                        "condition": "light rain",
                    },
                    {
                        "datetime": "2026-11-16 05:30:00",
                        "destination_local_datetime": "2026-11-16 11:00:00+05:30",
                        "temperature_c": 18.08,
                        "condition": "clear sky",
                    },
                ],
            },
            {"destination": "Jaipur"},
        )

        self.assertIn("**Current conditions:** 22°C and hazy.", rendered)
        self.assertIn("Temperatures around 18 to 31°C", rendered)
        self.assertIn("mostly clear", rendered)
        self.assertIn("light rain possible around Nov 15", rendered)
        self.assertNotIn("2026-11-12 23:30:00", rendered)
        self.assertLessEqual(len(rendered.splitlines()), 15)

    def test_weather_date_validation_rejects_invalid_and_conflicting_inputs(self):
        end, error = _resolve_weather_forecast_dates(
            "2026-02-30",
            "",
            "5 days",
        )
        self.assertEqual(end, "")
        self.assertIn("Travel date '2026-02-30' is invalid", error)

        _, error = _resolve_weather_forecast_dates(
            "2026-11-15",
            "2026-11-12",
            "",
        )
        self.assertIn("is before travel date", error)

        _, error = _resolve_weather_forecast_dates(
            "2026-11-10",
            "2026-11-12",
            "5 days",
        )
        self.assertIn("span 3 days", error)
        self.assertIn("return date would be 2026-11-14", error)

        end, error = _resolve_weather_forecast_dates(
            "2026-11-10",
            "",
            "5 days",
        )
        self.assertEqual(end, "2026-11-14")
        self.assertIsNone(error)

    def test_weather_agent_reports_date_conflict_instead_of_no_forecast(self):
        current_response = {
            "temperature_c": 25,
            "feels_like_c": 26,
            "humidity": 40,
            "condition": "clear",
            "wind_speed": 2,
            "timezone_offset_seconds": 19800,
        }
        forecast_response = {
            "timezone_offset_seconds": 19800,
            "forecast": [{
                "datetime": "2026-11-10 12:00:00",
                "temperature_c": 23,
                "condition": "clear",
            }],
        }

        with (
            patch(
                "backend.weather_mcp_search",
                new_callable=AsyncMock,
                return_value=current_response,
            ),
            patch(
                "backend.forecast_mcp_search",
                new_callable=AsyncMock,
                return_value=forecast_response,
            ),
        ):
            result = weather_agent({
                "trip_constraints": {
                    "destination": "Jaipur",
                    "travel_date": "2026-11-10",
                    "return_date": "2026-11-12",
                    "duration": "5 days",
                },
                "llm_calls": 0,
            })

        message = result["weather_data"]["forecast_date_matching_error"]
        self.assertIn("span 3 days", message)
        self.assertIn(message, result["weather_results"])
        rendered = render_weather_section(
            result["weather_data"],
            {
                "destination": "Jaipur",
                "travel_date": "2026-11-10",
                "return_date": "2026-11-12",
            },
        )
        self.assertIn(message, rendered)
        self.assertNotIn("No returned provider forecast entries", rendered)

    def test_get_forecast_returns_all_provider_entries(self):
        provider_entries = [
            {
                "dt_txt": f"2026-11-{12 + index // 8:02d} "
                f"{(index % 8) * 3:02d}:00:00",
                "main": {"temp": 20 + index},
                "weather": [{"description": "clear"}],
            }
            for index in range(12)
        ]
        provider_response = {
            "city": {
                "name": "Jaipur",
                "country": "IN",
                "coord": {"lat": 26.9124, "lon": 75.7873},
                "timezone": 19800,
            },
            "list": provider_entries,
        }

        with patch(
            "custom_weather_mcp_server._request_json",
            return_value=provider_response,
        ):
            result = get_forecast("Jaipur")

        self.assertEqual(len(result["forecast"]), 12)
        self.assertEqual(
            result["forecast"][-1]["datetime"],
            provider_entries[-1]["dt_txt"],
        )
        self.assertEqual(result["timezone_offset_seconds"], 19800)
        self.assertEqual(result["country_code"], "IN")
        self.assertEqual(result["latitude"], 26.9124)
        self.assertEqual(result["longitude"], 75.7873)

    def test_get_current_weather_returns_resolved_location_metadata(self):
        provider_response = {
            "name": "Paris",
            "sys": {"country": "FR"},
            "coord": {"lat": 48.8566, "lon": 2.3522},
            "main": {
                "temp": 20,
                "feels_like": 20,
                "humidity": 50,
            },
            "weather": [{"description": "clear"}],
            "wind": {"speed": 2},
            "timezone": 3600,
        }

        with patch(
            "custom_weather_mcp_server._request_json",
            return_value=provider_response,
        ):
            result = get_current_weather("Paris, Texas")

        self.assertEqual(result["city"], "Paris")
        self.assertEqual(result["country_code"], "FR")
        self.assertEqual(result["latitude"], 48.8566)
        self.assertEqual(result["longitude"], 2.3522)

    def test_weather_section_shows_requested_and_provider_resolved_locations(self):
        current_response = {
            "city": "Paris",
            "country_code": "FR",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "temperature_c": 20,
            "feels_like_c": 20,
            "humidity": 50,
            "condition": "clear",
            "wind_speed": 2,
            "timezone_offset_seconds": 3600,
        }
        forecast_response = {
            "city": "Paris",
            "country_code": "FR",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "timezone_offset_seconds": 3600,
            "forecast": [],
        }

        with (
            patch(
                "backend.weather_mcp_search",
                new_callable=AsyncMock,
                return_value=current_response,
            ),
            patch(
                "backend.forecast_mcp_search",
                new_callable=AsyncMock,
                return_value=forecast_response,
            ),
        ):
            result = weather_agent({
                "trip_constraints": {"destination": "Paris, Texas"},
                "llm_calls": 0,
            })

        rendered = render_weather_section(
            result["weather_data"],
            {"destination": "Paris, Texas"},
        )
        self.assertIn("**Requested destination:** Paris, Texas.", rendered)
        self.assertIn(
            "**Provider-resolved location:** Paris, FR (48.8566, 2.3522).",
            rendered,
        )
        self.assertTrue(result["weather_data"]["location_mismatch"])
        self.assertIn("Location mismatch", rendered)
        self.assertIn("withheld", rendered)
        self.assertNotIn("- Temperature: 20°C", rendered)

    def test_weather_agent_uses_iana_timezone_across_dst_transition(self):
        current_response = {
            "city": "New York",
            "country_code": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "temperature_c": 10,
            "feels_like_c": 10,
            "humidity": 50,
            "condition": "clear",
            "wind_speed": 2,
            "timezone_offset_seconds": -18000,
        }
        forecast_response = {
            "city": "New York",
            "country_code": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone_offset_seconds": -18000,
            "forecast": [{
                "datetime": "2026-03-08 07:30:00",
                "temperature_c": 8,
                "condition": "clear",
            }],
        }

        with (
            patch(
                "backend.weather_mcp_search",
                new_callable=AsyncMock,
                return_value=current_response,
            ),
            patch(
                "backend.forecast_mcp_search",
                new_callable=AsyncMock,
                return_value=forecast_response,
            ),
        ):
            result = weather_agent({
                "trip_constraints": {
                    "destination": "New York, USA",
                    "travel_date": "2026-03-08",
                    "return_date": "2026-03-08",
                },
                "llm_calls": 0,
            })

        weather_data = result["weather_data"]
        self.assertEqual(weather_data["timezone_name"], "America/New_York")
        self.assertEqual(
            weather_data["forecast_for_trip"][0][
                "destination_local_datetime"
            ],
            "2026-03-08 03:30:00-04:00",
        )
        self.assertIsNone(weather_data["timezone_resolution_warning"])

    def test_fallback_weather_routing_matches_words_not_substrings(self):
        self.assertNotIn(
            "weather_agent",
            fallback_agent_selection("Find a train to Boston."),
        )
        self.assertIn(
            "weather_agent",
            fallback_agent_selection("Is there rain in Boston?"),
        )

    def test_weather_agent_matches_utc_timestamp_to_destination_local_date(self):
        current_response = {
            "temperature_c": 25,
            "feels_like_c": 26,
            "humidity": 40,
            "condition": "clear",
            "wind_speed": 2,
            "timezone_offset_seconds": 19800,
        }
        forecast_response = {
            "timezone_offset_seconds": 19800,
            "forecast": [
                {
                    "datetime": "2026-11-12 20:00:00",
                    "temperature_c": 23,
                    "condition": "clear",
                },
                {
                    "datetime": "2026-11-12 17:00:00",
                    "temperature_c": 22,
                    "condition": "cloudy",
                },
            ],
        }

        with (
            patch(
                "backend.weather_mcp_search",
                new_callable=AsyncMock,
                return_value=current_response,
            ),
            patch(
                "backend.forecast_mcp_search",
                new_callable=AsyncMock,
                return_value=forecast_response,
            ),
        ):
            result = weather_agent({
                "trip_constraints": {
                    "destination": "Jaipur",
                    "travel_date": "2026-11-13",
                    "return_date": "2026-11-13",
                },
                "llm_calls": 0,
            })

        self.assertEqual(
            [item["datetime"] for item in result["weather_data"]["forecast_for_trip"]],
            ["2026-11-12 20:00:00"],
        )
        self.assertEqual(
            result["weather_data"]["forecast_for_trip"][0][
                "destination_local_datetime"
            ],
            "2026-11-13 01:30:00+05:30",
        )
        self.assertEqual(
            result["weather_data"]["timezone_offset_seconds"],
            19800,
        )
        self.assertIn("Available forecast (Nov 13)", result["weather_results"])
        self.assertIn("Temperatures around 23°C", result["weather_results"])
        self.assertNotIn("2026-11-13 01:30:00+05:30:", result["weather_results"])
        self.assertIsNone(
            result["weather_data"]["forecast_date_matching_error"]
        )

    def test_weather_agent_rejects_invalid_entries_sorts_and_deduplicates(self):
        current_response = {
            "temperature_c": 25,
            "feels_like_c": 26,
            "humidity": 40,
            "condition": "clear",
            "wind_speed": 2,
            "timezone_offset_seconds": 19800,
        }
        forecast_response = {
            "timezone_offset_seconds": 19800,
            "forecast": [
                {
                    "datetime": "2026-11-12 15:00:00",
                    "temperature_c": 23,
                    "condition": "cloudy",
                },
                {
                    "datetime": "2026-11-12T12:00:00Z",
                    "temperature_c": 22,
                    "condition": "clear ",
                },
                {
                    "datetime": "2026-11-12 17:30:00+05:30",
                    "temperature_c": 99,
                    "condition": "duplicate instant",
                },
                {
                    "datetime": "not-a-timestamp",
                    "temperature_c": 21,
                    "condition": "clear",
                },
                {
                    "datetime": "2026-11-12 18:00:00",
                    "temperature_c": "warm",
                    "condition": "clear",
                },
                {
                    "datetime": "2026-11-12 21:00:00",
                    "temperature_c": 20,
                    "condition": " ",
                },
            ],
        }

        with (
            patch(
                "backend.weather_mcp_search",
                new_callable=AsyncMock,
                return_value=current_response,
            ),
            patch(
                "backend.forecast_mcp_search",
                new_callable=AsyncMock,
                return_value=forecast_response,
            ),
        ):
            result = weather_agent({
                "trip_constraints": {"destination": "Jaipur"},
                "llm_calls": 0,
            })

        valid_forecast = result["weather_data"]["forecast"]
        self.assertEqual(
            [item["datetime"] for item in valid_forecast],
            ["2026-11-12T12:00:00Z", "2026-11-12 15:00:00"],
        )
        self.assertEqual(valid_forecast[0]["condition"], "clear")
        self.assertEqual(
            result["weather_data"]["forecast_validation"],
            {
                "invalid_entries_discarded": 3,
                "duplicates_removed": 1,
            },
        )

    def test_weather_section_summarizes_trip_forecast_without_hourly_entries(self):
        provider_forecast = [
            {
                "datetime": f"2026-11-{12 + index // 8:02d} "
                f"{(index % 8) * 3:02d}:00:00",
                "temperature_c": 20 + index,
                "condition": "clear",
            }
            for index in range(12)
        ]
        weather_data = {
            "city": "Jaipur",
            "current_weather": None,
            "forecast": provider_forecast,
            "forecast_for_trip": [provider_forecast[8]],
            "forecast_end": "2026-11-13",
        }

        rendered = render_weather_section(
            weather_data,
            {
                "destination": "Jaipur",
                "travel_date": "2026-11-13",
                "return_date": "2026-11-13",
            },
        )

        self.assertIn("**Forecast for your trip (available Nov 13):**", rendered)
        self.assertIn("Temperatures around 28°C", rendered)
        self.assertIn("clear", rendered)
        self.assertNotIn("2026-11-13 00:00:00:", rendered)


if __name__ == "__main__":
    unittest.main()
