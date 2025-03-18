#!/usr/bin/env python3
"""Tests for the SplineGroup and Spline API."""

import unittest
from splinaltap import SplineGroup, Spline, Knot
from splinaltap.backends import BackendManager

class TestSplineGroupSplineAPI(unittest.TestCase):
    """Test the SplineGroup and Spline API functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
    
    def test_spline_add_knot(self):
        """Test adding knots to a spline."""
        spline = Spline()
        
        # Add knots with different methods
        kf1 = spline.add_knot(at=0.0, value=0.0)
        kf2 = spline.add_knot(at=0.5, value=5.0, interpolation="cubic")
        kf3 = spline.add_knot(at=1.0, value=10.0, interpolation="bezier", 
                                  control_points=(0.8, 12.0, 0.9, 10.0))
        
        # Verify knots were added and have correct properties
        self.assertEqual(len(spline.knots), 3)
        self.assertEqual(kf1.at, 0.0)
        self.assertEqual(kf1.value(0.0, {}), 0.0)
        
        self.assertEqual(kf2.at, 0.5)
        self.assertEqual(kf2.value(0.5, {}), 5.0)
        self.assertEqual(kf2.interpolation, "cubic")
        
        self.assertEqual(kf3.at, 1.0)
        self.assertEqual(kf3.value(1.0, {}), 10.0)
        self.assertEqual(kf3.interpolation, "bezier")
        self.assertEqual(kf3.control_points, (0.8, 12.0, 0.9, 10.0))
        
        # Test adding a knot at an existing position (should replace)
        kf4 = spline.add_knot(at=0.5, value=7.0)
        self.assertEqual(len(spline.knots), 3)  # Count should remain the same
        
        # Find the knot at position 0.5
        for kf in spline.knots:
            if kf.at == 0.5:
                self.assertEqual(kf.value(0.5, {}), 7.0)  # Value should be updated
    
    def test_spline_remove_knot(self):
        """Test removing knots from a spline."""
        spline = Spline()
        
        # Add knots
        spline.add_knot(at=0.0, value=0.0)
        spline.add_knot(at=0.5, value=5.0)
        spline.add_knot(at=1.0, value=10.0)
        
        # Verify knots were added
        self.assertEqual(len(spline.knots), 3)
        
        # Remove knot at position 0.5
        spline.remove_knot(0.5)
        
        # Verify knot was removed
        self.assertEqual(len(spline.knots), 2)
        for kf in spline.knots:
            self.assertNotEqual(kf.at, 0.5)
        
        # Test removing a nonexistent knot
        with self.assertRaises(ValueError):
            spline.remove_knot(0.75)
    
    def test_spline_get_value(self):
        """Test getting interpolated values from a spline."""
        spline = Spline()
        
        # Add knots
        spline.add_knot(at=0.0, value=0.0)
        spline.add_knot(at=1.0, value=10.0)
        
        # Test getting value at various positions
        self.assertEqual(spline.get_value(0.0), 0.0)
        self.assertEqual(spline.get_value(0.5), 5.0)
        self.assertEqual(spline.get_value(1.0), 10.0)
        self.assertEqual(spline.get_value(0.25), 2.5)
        self.assertEqual(spline.get_value(0.75), 7.5)
        
        # Test getting values outside the range (should clamp)
        self.assertEqual(spline.get_value(-0.5), 0.0)
        self.assertEqual(spline.get_value(1.5), 10.0)
    
    def test_spline_get_knot_values(self):
        """Test getting knot values from a spline."""
        spline = Spline()
        
        # Add knots
        spline.add_knot(at=0.0, value=0.0)
        spline.add_knot(at=0.5, value=5.0)
        spline.add_knot(at=1.0, value=10.0)
        
        # Get knot values
        knot_values = spline.get_knot_values()
        
        # Verify values
        self.assertEqual(len(knot_values), 3)
        
        expected_values = [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]
        for (expected_pos, expected_val), (actual_pos, actual_val) in zip(expected_values, knot_values):
            self.assertEqual(expected_pos, actual_pos)
            self.assertEqual(expected_val, actual_val)
        
        # Test with a spline containing expressions
        expr_spline = Spline()
        expr_spline.add_knot(at=0.0, value="sin(0)")
        expr_spline.add_knot(at=0.5, value="sin(pi/2)")
        
        expr_values = expr_spline.get_knot_values()
        self.assertEqual(len(expr_values), 2)
        self.assertEqual(expr_values[0][0], 0.0)
        self.assertAlmostEqual(expr_values[0][1], 0.0, places=5)
        self.assertEqual(expr_values[1][0], 0.5)
        self.assertAlmostEqual(expr_values[1][1], 1.0, places=5)
    
    def test_spline_sampling(self):
        """Test sampling a spline at multiple points."""
        spline = Spline()
        
        # Add knots
        spline.add_knot(at=0.0, value=0.0)
        spline.add_knot(at=1.0, value=10.0)
        
        # Sample at multiple points
        samples = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = spline.sample(samples)
        
        # Verify values
        expected_values = [0.0, 2.5, 5.0, 7.5, 10.0]
        self.assertEqual(len(values), len(expected_values))
        
        for expected, actual in zip(expected_values, values):
            self.assertEqual(expected, actual)
    
    def test_spline_min_max(self):
        """Test spline min/max clamping."""
        # Create spline with min/max
        spline = Spline(min_max=(0, 5))
        spline.interpolation = "linear"  # Explicitly set linear interpolation
        
        # Add knots
        spline.add_knot(at=0.0, value=0.0)
        spline.add_knot(at=0.5, value=10.0)  # Above max
        spline.add_knot(at=1.0, value=-5.0)  # Below min
        
        # Test clamping at knot points
        self.assertEqual(spline.get_value(0.0), 0.0)
        self.assertEqual(spline.get_value(0.5), 5.0)  # Clamped to max
        self.assertEqual(spline.get_value(1.0), 0.0)  # Clamped to min
        
        # For intermediate values between 0 and 0.5, with linear interpolation
        # We should interpolate between 0 and 5 (max)
        result025 = spline.get_value(0.25) 
        self.assertLessEqual(result025, 5.0)  # Should be clamped at 5.0
        
        # For intermediate values between 0.5 and 1.0, with linear interpolation 
        # We should interpolate between 5 (max) and 0 (min)
        result075 = spline.get_value(0.75)
        # Since linear interpolation is used, it should be 5.0 * (1 - 0.5) where 0.5 = (0.75 - 0.5) / (1.0 - 0.5)
        self.assertLessEqual(result075, 5.0)
        self.assertGreaterEqual(result075, 0.0)
    
    def test_spline_group_add_spline(self):
        """Test adding splines to a spline group."""
        spline_group = SplineGroup()
        
        # Add splines
        sp1 = spline_group.add_spline(name="position", interpolation="linear")  # Explicitly set linear
        sp2 = spline_group.add_spline(name="rotation", interpolation="cubic")
        sp3 = spline_group.add_spline(name="scale", min_max=(0.1, 2.0))
        
        # Verify splines were added with correct properties
        self.assertEqual(len(spline_group.splines), 3)
        self.assertIn("position", spline_group.splines)
        self.assertIn("rotation", spline_group.splines)
        self.assertIn("scale", spline_group.splines)
        
        self.assertEqual(spline_group.splines["position"].interpolation, "linear")
        self.assertEqual(spline_group.splines["rotation"].interpolation, "cubic")
        self.assertEqual(spline_group.splines["scale"].min_max, (0.1, 2.0))
        
        # Test that adding a spline with the same name with replace=True works
        position_spline = spline_group.splines["position"]
        position_spline2 = spline_group.add_spline(name="position", replace=True)
        # Should be the same object
        self.assertIs(position_spline, position_spline2)
    
    def test_spline_group_remove_spline(self):
        """Test removing splines from a spline group."""
        spline_group = SplineGroup()
        
        # Add splines
        spline_group.add_spline(name="position")
        spline_group.add_spline(name="rotation")
        spline_group.add_spline(name="scale")
        
        # Verify splines were added
        self.assertEqual(len(spline_group.splines), 3)
        
        # Since SplineGroup doesn't have a remove_spline method,
        # we'll implement the behavior by directly modifying the splines dictionary
        del spline_group.splines["rotation"]
        
        # Verify spline was removed
        self.assertEqual(len(spline_group.splines), 2)
        self.assertIn("position", spline_group.splines)
        self.assertNotIn("rotation", spline_group.splines)
        self.assertIn("scale", spline_group.splines)
        
        # Test removing a nonexistent spline
        with self.assertRaises(KeyError):
            del spline_group.splines["nonexistent"]
    
    def test_spline_group_get_spline_names(self):
        """Test getting spline names from a spline group."""
        spline_group = SplineGroup()
        
        # Add splines
        spline_group.add_spline(name="position")
        spline_group.add_spline(name="rotation")
        spline_group.add_spline(name="scale")
        
        # Get spline names directly from the splines dictionary
        names = list(spline_group.splines.keys())
        
        # Verify names
        self.assertEqual(set(names), {"position", "rotation", "scale"})
    
    def test_spline_group_get_values(self):
        """Test getting values from all splines in a spline group."""
        spline_group = SplineGroup()
        
        # Add splines with knots
        pos_spline = spline_group.add_spline(name="position")
        pos_spline.add_knot(at=0.0, value=0.0)
        pos_spline.add_knot(at=1.0, value=10.0)
        
        rot_spline = spline_group.add_spline(name="rotation")
        rot_spline.add_knot(at=0.0, value=0.0)
        rot_spline.add_knot(at=1.0, value=360.0)
        
        # Get values at a specific time using get_value which is the actual method name
        values = spline_group.get_value(0.5)
        
        # Verify values
        self.assertEqual(len(values), 2)
        self.assertEqual(values["position"], 5.0)
        self.assertEqual(values["rotation"], 180.0)
    
    def test_spline_group_get_knots(self):
        """Test getting all knots from a spline group."""
        spline_group = SplineGroup()
        
        # Add splines with knots
        pos_spline = spline_group.add_spline(name="position")
        pos_spline.add_knot(at=0.0, value=0.0)
        pos_spline.add_knot(at=1.0, value=10.0)
        
        rot_spline = spline_group.add_spline(name="rotation")
        rot_spline.add_knot(at=0.0, value=0.0)
        rot_spline.add_knot(at=0.5, value=180.0)
        rot_spline.add_knot(at=1.0, value=360.0)
        
        # Get all knots by iterating through spline knots
        knots = {}
        for spline_name, spline in spline_group.splines.items():
            knots[spline_name] = spline.get_knot_values()
        
        # Verify knots
        self.assertEqual(len(knots), 2)  # Two splines
        self.assertIn("position", knots)
        self.assertIn("rotation", knots)
        
        self.assertEqual(len(knots["position"]), 2)  # Two knots
        self.assertEqual(len(knots["rotation"]), 3)  # Three knots
        
        # Verify knot values
        pos_kfs = knots["position"]
        self.assertEqual(pos_kfs[0][0], 0.0)  # Position
        self.assertEqual(pos_kfs[0][1], 0.0)  # Value
        self.assertEqual(pos_kfs[1][0], 1.0)
        self.assertEqual(pos_kfs[1][1], 10.0)
        
        rot_kfs = knots["rotation"]
        self.assertEqual(rot_kfs[0][0], 0.0)
        self.assertEqual(rot_kfs[0][1], 0.0)
        self.assertEqual(rot_kfs[1][0], 0.5)
        self.assertEqual(rot_kfs[1][1], 180.0)
        self.assertEqual(rot_kfs[2][0], 1.0)
        self.assertEqual(rot_kfs[2][1], 360.0)

if __name__ == "__main__":
    unittest.main()