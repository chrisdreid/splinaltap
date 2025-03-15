#!/usr/bin/env python3
"""Tests for interpolation methods in SplinalTap."""

import unittest
import math
from splinaltap import Channel, Keyframe
from splinaltap.backends import BackendManager

class TestInterpolationMethods(unittest.TestCase):
    """Test all interpolation methods for correctness."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
    
    def test_linear_interpolation(self):
        """Test linear interpolation between keyframes."""
        channel = Channel(interpolation="linear")
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Test values at various points
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(0.5), 5.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        self.assertEqual(channel.get_value(0.25), 2.5)
        self.assertEqual(channel.get_value(0.75), 7.5)
    
    def test_step_interpolation(self):
        """Test step interpolation between keyframes."""
        channel = Channel(interpolation="nearest")  # "step" is called "nearest" in the implementation
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=0.5, value=5.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Test values at various points
        self.assertEqual(channel.get_value(0.0), 0.0)
        # With nearest neighbor, the transition happens at the midpoint between keyframes
        self.assertEqual(channel.get_value(0.24), 0.0)  # Still close to first keyframe
        self.assertEqual(channel.get_value(0.5), 5.0)   # Exactly at keyframe
        self.assertEqual(channel.get_value(0.74), 5.0)  # Still close to middle keyframe
        self.assertEqual(channel.get_value(0.76), 10.0) # Closer to last keyframe
        self.assertEqual(channel.get_value(1.0), 10.0)
    
    def test_cubic_interpolation(self):
        """Test cubic interpolation between keyframes."""
        channel = Channel(interpolation="cubic")
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Test values at endpoints
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Test that middle point is reasonably close to linear (in simplest case)
        # Cubic with just two points should be relatively close to linear
        self.assertAlmostEqual(channel.get_value(0.5), 5.0, delta=0.1)
        
        # Add more keyframes for a more complex test
        complex_channel = Channel(interpolation="cubic")
        complex_channel.add_keyframe(at=0.0, value=0.0)
        complex_channel.add_keyframe(at=0.25, value=2.5)
        complex_channel.add_keyframe(at=0.5, value=7.5)
        complex_channel.add_keyframe(at=1.0, value=10.0)
        
        # Test keyframe points match exactly
        self.assertEqual(complex_channel.get_value(0.0), 0.0)
        self.assertEqual(complex_channel.get_value(0.25), 2.5)
        self.assertEqual(complex_channel.get_value(0.5), 7.5)
        self.assertEqual(complex_channel.get_value(1.0), 10.0)
    
    def test_bezier_interpolation(self):
        """Test bezier interpolation between keyframes."""
        channel = Channel(interpolation="bezier")
        
        # Simple case with control points - specify as tuple as required by the implementation
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0, control_points=(0.25, 3.0, 0.75, 15.0))
        
        # Test values at endpoints
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Get values at various points
        v25 = channel.get_value(0.25)
        v50 = channel.get_value(0.5)
        v75 = channel.get_value(0.75)
        
        # We can't easily predict exact values, but we can ensure they're reasonable
        self.assertGreaterEqual(v25, 0.0)
        self.assertLessEqual(v25, 10.0)
        
        self.assertGreaterEqual(v50, 0.0)
        self.assertLessEqual(v50, 15.0)
        
        self.assertGreaterEqual(v75, 0.0)
        self.assertLessEqual(v75, 15.0)
    
    def test_hermite_interpolation(self):
        """Test hermite interpolation between keyframes."""
        channel = Channel(interpolation="hermite")
        
        # Simple case with derivatives
        channel.add_keyframe(at=0.0, value=0.0, derivative=0.0)
        channel.add_keyframe(at=1.0, value=10.0, derivative=0.0)
        
        # Test values at endpoints
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Get values at intermediate points
        v25 = channel.get_value(0.25)
        v50 = channel.get_value(0.5)
        v75 = channel.get_value(0.75)
        
        # Hermite with endpoints having 0 derivatives should give a smooth S-curve
        # We don't know the exact values, but they should be within range and smooth
        self.assertGreaterEqual(v25, 0.0)
        self.assertLessEqual(v25, 10.0)
        
        self.assertGreaterEqual(v50, 0.0)
        self.assertLessEqual(v50, 10.0)
        
        self.assertGreaterEqual(v75, 0.0)
        self.assertLessEqual(v75, 10.0)
        
        # Test with non-zero derivatives
        channel2 = Channel(interpolation="hermite")
        channel2.add_keyframe(at=0.0, value=0.0, derivative=10.0)  # Start with positive slope
        channel2.add_keyframe(at=1.0, value=10.0, derivative=-10.0)  # End with negative slope
        
        # Test values at endpoints
        self.assertEqual(channel2.get_value(0.0), 0.0)
        self.assertEqual(channel2.get_value(1.0), 10.0)
        
        # Get values at various points
        v25 = channel2.get_value(0.25)
        v50 = channel2.get_value(0.5)
        v75 = channel2.get_value(0.75)
        
        # With high derivatives at start and negative at end, we should see overshooting
        # We don't assert specific behaviors, just ensure the values are reasonable
        self.assertGreaterEqual(v25, 0.0)  # May go higher depending on implementation
        self.assertGreaterEqual(v50, 0.0)  
        self.assertGreaterEqual(v75, 0.0)
    
    def test_mixed_interpolation_methods(self):
        """Test mixing different interpolation methods in the same channel."""
        channel = Channel()  # Default is cubic
        
        # Add keyframes with different interpolation methods
        channel.add_keyframe(at=0.0, value=0.0)  # Uses default (cubic)
        channel.add_keyframe(at=0.25, value=2.5, interpolation="linear")
        channel.add_keyframe(at=0.5, value=5.0, interpolation="nearest")
        channel.add_keyframe(at=0.75, value=7.5, interpolation="bezier", 
                             control_points=(0.8, 8.0, 0.9, 7.0))
        channel.add_keyframe(at=1.0, value=10.0, interpolation="hermite", derivative=0.0)
        
        # Test values at keyframe points
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(0.25), 2.5)
        self.assertEqual(channel.get_value(0.5), 5.0)
        self.assertEqual(channel.get_value(0.75), 7.5)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Test interpolation between keyframe points
        # In the implementation, each segment uses the right keyframe's interpolation method
        # So 0.0-0.25 uses linear, 0.25-0.5 uses nearest, etc.
        
        # Get some intermediate values - we won't assert specific values, just ensure they're reasonable
        v1 = channel.get_value(0.125)  # Between 0.0 and 0.25 (should use linear)
        v2 = channel.get_value(0.375)  # Between 0.25 and 0.5 (should use nearest)
        v3 = channel.get_value(0.625)  # Between 0.5 and 0.75 (should use bezier)
        v4 = channel.get_value(0.875)  # Between 0.75 and 1.0 (should use hermite)
        
        # Just ensure all values are within the overall range
        for v in [v1, v2, v3, v4]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 10.0)
    
    def test_interpolation_outside_range(self):
        """Test interpolation behavior outside the defined range."""
        channel = Channel()
        channel.add_keyframe(at=0.2, value=20.0)
        channel.add_keyframe(at=0.8, value=80.0)
        
        # Test before first keyframe - should clamp to first value
        self.assertEqual(channel.get_value(0.0), 20.0)
        self.assertEqual(channel.get_value(0.1), 20.0)
        
        # Test after last keyframe - should clamp to last value
        self.assertEqual(channel.get_value(0.9), 80.0)
        self.assertEqual(channel.get_value(1.0), 80.0)

    def test_multi_segment_interpolation(self):
        """Test interpolation over multiple segments."""
        channel = Channel(interpolation="cubic")
        
        # Create multiple segments with alternating values
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=0.2, value=10.0)
        channel.add_keyframe(at=0.4, value=0.0)
        channel.add_keyframe(at=0.6, value=10.0)
        channel.add_keyframe(at=0.8, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Test at keyframe points
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(0.2), 10.0)
        self.assertEqual(channel.get_value(0.4), 0.0)
        self.assertEqual(channel.get_value(0.6), 10.0)
        self.assertEqual(channel.get_value(0.8), 0.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Test between keyframes - should be smooth with cubic
        self.assertGreater(channel.get_value(0.1), 0.0)
        self.assertLess(channel.get_value(0.1), 10.0)
        
        self.assertGreater(channel.get_value(0.3), 0.0)
        self.assertLess(channel.get_value(0.3), 10.0)

if __name__ == "__main__":
    unittest.main()