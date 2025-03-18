#!/usr/bin/env python3
"""Tests for core SplinalTap objects."""

import unittest
import math
from splinaltap import SplineSolver, SplineGroup, Spline, Knot
from splinaltap.backends import BackendManager

class TestCoreObjects(unittest.TestCase):
    """Test core object instantiation and basic functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
    
    def test_keyframe_instantiation(self):
        """Test that Keyframe objects can be properly instantiated."""
        # Test with numeric value
        kf1 = Knot(at=0.5, value=lambda t, ctx: 10.0)
        self.assertEqual(kf1.at, 0.5)
        
        # Test with interpolation method
        kf3 = Knot(at=0.25, value=lambda t, ctx: 5.0, interpolation="cubic")
        self.assertEqual(kf3.at, 0.25)
        self.assertEqual(kf3.interpolation, "cubic")
        
        # Test with control points for Bezier
        kf4 = Knot(at=0.75, value=lambda t, ctx: 15.0, interpolation="bezier", 
                 control_points=[0.8, 16.0, 0.9, 14.0])
        self.assertEqual(kf4.at, 0.75)
        self.assertEqual(kf4.interpolation, "bezier")
        self.assertEqual(kf4.control_points, [0.8, 16.0, 0.9, 14.0])
        
        # Test with derivative for Hermite
        kf5 = Knot(at=1.0, value=lambda t, ctx: 20.0, interpolation="hermite", derivative=0.0)
        self.assertEqual(kf5.at, 1.0)
        self.assertEqual(kf5.interpolation, "hermite")
        self.assertEqual(kf5.derivative, 0.0)
    
    def test_channel_instantiation(self):
        """Test that Channel objects can be properly instantiated."""
        # Test basic instantiation
        channel = Spline()
        self.assertEqual(len(channel.knots), 0)
        self.assertEqual(channel.interpolation, "cubic")  # Default interpolation
        
        # Test with custom interpolation
        channel2 = Spline(interpolation="linear")
        self.assertEqual(channel2.interpolation, "linear")
        
        # Test with min/max values
        channel3 = Spline(min_max=(0, 100))
        self.assertEqual(channel3.min_max, (0, 100))
    
    def test_spline_instantiation(self):
        """Test that Spline objects can be properly instantiated."""
        # Test basic instantiation
        spline_group = SplineGroup()  # SplineGroup is the new Spline
        self.assertEqual(len(spline_group.splines), 0)
        
        # Test add_spline method (formerly add_channel)
        spline = spline_group.add_spline(name="position")
        self.assertIn("position", spline_group.splines)
        self.assertEqual(spline_group.splines["position"], spline)
        
        # Test add_spline with parameters
        spline2 = spline_group.add_spline(name="rotation", interpolation="cubic", min_max=(-180, 180))
        self.assertIn("rotation", spline_group.splines)
        self.assertEqual(spline2.interpolation, "cubic")
        self.assertEqual(spline2.min_max, (-180, 180))
        
        # Test get_spline method (formerly get_channel)
        self.assertEqual(spline_group.get_spline("position"), spline)
        self.assertEqual(spline_group.get_spline("rotation"), spline2)
        
        # Test get_spline with nonexistent spline
        with self.assertRaises(ValueError):
            spline_group.get_spline("nonexistent")
    
    def test_solver_instantiation(self):
        """Test that KeyframeSolver objects can be properly instantiated."""
        # Test basic instantiation
        solver = SplineSolver(name="test_solver")
        self.assertEqual(solver.name, "test_solver")
        self.assertEqual(len(solver.spline_groups), 0)
        self.assertEqual(solver.range, (0.0, 1.0))  # Default range
        
        # Test setting range after instantiation
        solver2 = SplineSolver(name="custom_range")
        solver2.range = (0, 100)
        self.assertEqual(solver2.range, (0, 100))
        
        # Test create_spline method (now create_spline_group)
        spline_group = solver.create_spline_group(name="main")
        self.assertIn("main", solver.spline_groups)
        self.assertEqual(solver.spline_groups["main"], spline_group)
        
        # Test get_spline method (now get_spline_group)
        self.assertEqual(solver.get_spline_group("main"), spline_group)
        
        # Test get_spline with nonexistent spline
        with self.assertRaises(KeyError):
            solver.get_spline_group("nonexistent")
        
        # Test setting metadata directly
        solver.metadata["author"] = "Test User"
        self.assertEqual(solver.metadata["author"], "Test User")
        
        # Test variables
        solver.set_variable("pi", 3.14159)
        self.assertAlmostEqual(solver.variables["pi"], 3.14159)

    def test_object_relationships(self):
        """Test that objects can be properly connected and related."""
        # Create a complete hierarchy
        solver = SplineSolver(name="test_solver")
        spline_group = solver.create_spline_group(name="motion")
        
        # Add value channel for backward compatibility
        value_channel = spline_group.add_spline("value")
        value_channel.add_knot(at=0.0, value=0.0)
        value_channel.add_knot(at=1.0, value=1.0)
        
        spline1 = spline_group.add_spline(name="position")
        spline2 = spline_group.add_spline(name="velocity")
        
        # Add keyframes to splines (formerly channels)
        spline1.add_knot(at=0.0, value=0.0)
        spline1.add_knot(at=1.0, value=10.0)
        
        spline2.add_knot(at=0.0, value=5.0)
        spline2.add_knot(at=1.0, value=0.0)
        
        # Verify relationships
        self.assertIn("motion", solver.spline_groups)
        self.assertIn("position", spline_group.splines)
        self.assertIn("velocity", spline_group.splines)
        self.assertEqual(len(spline1.knots), 2)
        self.assertEqual(len(spline2.knots), 2)
        
        # Test solve method
        result = solver.solve(0.5)
        
        # Verify result structure
        self.assertIn("motion", result)
        self.assertIn("position", result["motion"])
        self.assertIn("velocity", result["motion"])
        
        # Verify interpolated values at t=0.5
        self.assertAlmostEqual(result["motion"]["position"], 5.0)  # Linear interpolation
        self.assertAlmostEqual(result["motion"]["velocity"], 2.5)  # Linear interpolation

if __name__ == "__main__":
    unittest.main()