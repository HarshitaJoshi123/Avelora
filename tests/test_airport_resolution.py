import unittest
from unittest.mock import patch

from backend import get_airport_code


class AirportResolutionTests(unittest.TestCase):
    def test_multiple_city_airports_choose_nearest_to_city_center(self):
        # Coordinates near central Ludhiana are closest to LUH in airportsdata.
        with patch(
            "backend._get_city_center_coordinates",
            return_value=(30.9010, 75.8573),
        ):
            self.assertEqual(get_airport_code("Ludhiana"), "LUH")

    def test_single_airport_city_does_not_need_geocoding(self):
        with patch("backend._get_city_center_coordinates") as geocode:
            self.assertEqual(get_airport_code("Mumbai"), "BOM")
        geocode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
