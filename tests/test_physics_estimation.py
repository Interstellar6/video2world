import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("physics_estimation", Path(__file__).resolve().parents[1] / "scripts/physics_estimation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhysicsEstimateTest(unittest.TestCase):
    def test_unknown_dimensions_are_explicit_and_intervals_remain_intervals(self):
        value = MODULE.parse_estimate('{"length_m":null,"width_m":[1,2],"height_m":[0.4,0.8],"mass_kg":[40,100],"surface_smoothness":0.6,"confidence":0.3,"basis":"category prior"}')
        self.assertIsNone(value["length_m"])
        self.assertEqual(value["mass_kg"], [40, 100])

    def test_invalid_mass_or_smoothness_is_rejected(self):
        import json
        value = {"length_m": None, "width_m": None, "height_m": None, "mass_kg": [-1, 10],
                 "surface_smoothness": 0.3, "confidence": 0.5, "basis": "category prior"}
        with self.assertRaisesRegex(ValueError, "mass_kg"):
            MODULE.parse_estimate(json.dumps(value))
        value.update(mass_kg=[10, 20], surface_smoothness=5)
        with self.assertRaisesRegex(ValueError, "smoothness"):
            MODULE.parse_estimate(json.dumps(value))
