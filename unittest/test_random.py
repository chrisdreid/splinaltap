#!/usr/bin/env python3
"""Unit tests for random functions in SplinalTap."""

import unittest
import math

from splinaltap import KeyframeSolver, Spline, Channel
from splinaltap.backends import BackendManager

class TestRandomFunctions(unittest.TestCase):
    """Test the rand() and randint() functions in expressions."""

    def setUp(self):
        """Set up test case environment."""
        # Force Python backend for consistent results
        BackendManager.set_backend("python")
        self.channel = Channel()  # Use Channel instead of KeyframeInterpolator

    def test_rand_float_range(self):
        """Test that rand() returns values in the 0-1 range."""
        self.channel.add_keyframe(at=0.0, value="rand()")
        
        for _ in range(10):
            value = self.channel.get_value(0.0)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)
            self.assertIsInstance(value, float)

    def test_rand_scaling(self):
        """Test that rand() can be scaled to other ranges."""
        self.channel.add_keyframe(at=0.0, value="rand() * 10")
        
        for _ in range(10):
            value = self.channel.get_value(0.0)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 10.0)
            self.assertIsInstance(value, float)

    def test_randint_single_arg(self):
        """Test randint(max) returns integers between 0 and max."""
        self.channel.add_keyframe(at=0.0, value="randint(5)")
        
        values_seen = set()
        for _ in range(20):  # Run more iterations to likely see different values
            value = self.channel.get_value(0.0)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 5.0)
            self.assertEqual(value, math.floor(value))  # Check it's an integer
            values_seen.add(value)
        
        # Should have seen multiple different values
        self.assertGreater(len(values_seen), 1)

    def test_randint_range(self):
        """Test randint([min, max]) returns integers in the specified range."""
        self.channel.add_keyframe(at=0.0, value="randint([3, 7])")
        
        values_seen = set()
        for _ in range(20):  # Run more iterations to likely see different values
            value = self.channel.get_value(0.0)
            self.assertGreaterEqual(value, 3.0)
            self.assertLessEqual(value, 7.0)
            self.assertEqual(value, math.floor(value))  # Check it's an integer
            values_seen.add(value)
        
        # Should have seen multiple different values
        self.assertGreater(len(values_seen), 1)

    def test_random_with_solver(self):
        """Test random functions through the full KeyframeSolver architecture."""
        solver = KeyframeSolver()
        spline = solver.create_spline("noise")
        
        # Create white noise channel (random float)
        white_noise = spline.add_channel("white")
        white_noise.add_keyframe(at=0.0, value="rand() * 2 - 1")  # Range: -1 to 1
        white_noise.add_keyframe(at=1.0, value="rand() * 2 - 1")
        
        # Create integer-only random noise channel (directly using randint)
        int_noise = spline.add_channel("integer")
        int_noise.add_keyframe(at=0.0, value="randint([1, 5])")  # Integer values 1-5
        int_noise.add_keyframe(at=1.0, value="randint([1, 5])")
        
        # Test channel values at exact keyframe positions (to avoid interpolation)
        for _ in range(5):
            # Test at 0.0 (exact keyframe position)
            result = solver.solve(0.0)
            white_val = result["noise"]["white"]
            int_val = result["noise"]["integer"]
            
            # White noise should be between -1 and 1
            self.assertGreaterEqual(white_val, -1.0)
            self.assertLessEqual(white_val, 1.0)
            self.assertIsInstance(white_val, float)
            
            # Integer noise should be between 1 and 5 and an integer value
            self.assertIsInstance(int_val, float)  # Still returns as float type
            self.assertGreaterEqual(int_val, 1.0)
            self.assertLessEqual(int_val, 5.0)
            self.assertEqual(int_val, round(int_val), 
                             f"Value {int_val} is not an integer")

if __name__ == "__main__":
    unittest.main()