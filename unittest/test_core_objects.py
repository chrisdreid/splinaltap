#!/usr/bin/env python3
"""Tests for core SplinalTap objects."""

import unittest
import math
from splinaltap import KeyframeSolver, Spline, Channel, Keyframe
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
        kf1 = Keyframe(at=0.5, value=10.0)
        self.assertEqual(kf1.at, 0.5)
        # Value is stored directly, not as a callable in the test
        self.assertEqual(kf1.value, 10.0)
        
        # Test with expression
        # For expressions, we'd need the evaluator to parse them first
        # Skip testing expression evaluation here as it's covered in expression tests
        kf2 = Keyframe(at=0.0, value="sin(t)")
        self.assertEqual(kf2.at, 0.0)
        self.assertEqual(kf2.value, "sin(t)")
        
        # Test with interpolation method
        kf3 = Keyframe(at=0.25, value=5.0, interpolation="cubic")
        self.assertEqual(kf3.at, 0.25)
        self.assertEqual(kf3.value, 5.0)
        self.assertEqual(kf3.interpolation, "cubic")
        
        # Test with control points for Bezier
        kf4 = Keyframe(at=0.75, value=15.0, interpolation="bezier", 
                      control_points=(0.8, 16.0, 0.9, 14.0))
        self.assertEqual(kf4.at, 0.75)
        self.assertEqual(kf4.value, 15.0)
        self.assertEqual(kf4.interpolation, "bezier")
        self.assertEqual(kf4.control_points, (0.8, 16.0, 0.9, 14.0))
        
        # Test with derivative for Hermite
        kf5 = Keyframe(at=1.0, value=20.0, interpolation="hermite", derivative=0.0)
        self.assertEqual(kf5.at, 1.0)
        self.assertEqual(kf5.value, 20.0)
        self.assertEqual(kf5.interpolation, "hermite")
        self.assertEqual(kf5.derivative, 0.0)
    
    def test_channel_instantiation(self):
        """Test that Channel objects can be properly instantiated."""
        # Test basic instantiation
        channel = Channel()
        self.assertEqual(len(channel.keyframes), 0)
        self.assertEqual(channel.interpolation, "cubic")  # Default interpolation
        
        # Test with custom interpolation
        channel2 = Channel(interpolation="linear")
        self.assertEqual(channel2.interpolation, "linear")
        
        # Test with min/max values
        channel3 = Channel(min_max=(0, 100))
        self.assertEqual(channel3.min_max, (0, 100))
    
    def test_spline_instantiation(self):
        """Test that Spline objects can be properly instantiated."""
        # Test basic instantiation
        spline = Spline()  # Spline doesn't take a name parameter
        self.assertEqual(len(spline.channels), 0)
        
        # Test add_channel method
        channel = spline.add_channel(name="position")
        self.assertIn("position", spline.channels)
        self.assertEqual(spline.channels["position"], channel)
        
        # Test add_channel with parameters
        channel2 = spline.add_channel(name="rotation", interpolation="cubic", min_max=(-180, 180))
        self.assertIn("rotation", spline.channels)
        self.assertEqual(channel2.interpolation, "cubic")
        self.assertEqual(channel2.min_max, (-180, 180))
        
        # Test get_channel method
        self.assertEqual(spline.get_channel("position"), channel)
        self.assertEqual(spline.get_channel("rotation"), channel2)
        
        # Test get_channel with nonexistent channel
        with self.assertRaises(ValueError):  # Error handling in Spline.get_channel raises ValueError
            spline.get_channel("nonexistent")
    
    def test_solver_instantiation(self):
        """Test that KeyframeSolver objects can be properly instantiated."""
        # Test basic instantiation
        solver = KeyframeSolver(name="test_solver")
        self.assertEqual(solver.name, "test_solver")
        self.assertEqual(len(solver.splines), 0)
        self.assertEqual(solver.range, (0.0, 1.0))  # Default range
        
        # Test setting range after instantiation
        solver2 = KeyframeSolver(name="custom_range")
        solver2.range = (0, 100)
        self.assertEqual(solver2.range, (0, 100))
        
        # Test create_spline method
        spline = solver.create_spline(name="main")
        self.assertIn("main", solver.splines)
        self.assertEqual(solver.splines["main"], spline)
        
        # Test get_spline method
        self.assertEqual(solver.get_spline("main"), spline)
        
        # Test get_spline with nonexistent spline
        with self.assertRaises(KeyError):
            solver.get_spline("nonexistent")
        
        # Test setting metadata directly
        solver.metadata["author"] = "Test User"
        self.assertEqual(solver.metadata["author"], "Test User")
        
        # Test variables
        solver.set_variable("pi", 3.14159)
        self.assertAlmostEqual(solver.variables["pi"], 3.14159)

    def test_object_relationships(self):
        """Test that objects can be properly connected and related."""
        # Create a complete hierarchy
        solver = KeyframeSolver(name="test_solver")
        spline = solver.create_spline(name="motion")
        channel1 = spline.add_channel(name="position")
        channel2 = spline.add_channel(name="velocity")
        
        # Add keyframes to channels
        channel1.add_keyframe(at=0.0, value=0.0)
        channel1.add_keyframe(at=1.0, value=10.0)
        
        channel2.add_keyframe(at=0.0, value=5.0)
        channel2.add_keyframe(at=1.0, value=0.0)
        
        # Verify relationships
        self.assertIn("motion", solver.splines)
        self.assertIn("position", spline.channels)
        self.assertIn("velocity", spline.channels)
        self.assertEqual(len(channel1.keyframes), 2)
        self.assertEqual(len(channel2.keyframes), 2)
        
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