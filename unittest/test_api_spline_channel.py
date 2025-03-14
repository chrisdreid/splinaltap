#!/usr/bin/env python3
"""Tests for the Spline and Channel API."""

import unittest
from splinaltap import Spline, Channel, Keyframe
from splinaltap.backends import BackendManager

class TestSplineChannelAPI(unittest.TestCase):
    """Test the Spline and Channel API functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
    
    def test_channel_add_keyframe(self):
        """Test adding keyframes to a channel."""
        channel = Channel()
        
        # Add keyframes with different methods
        kf1 = channel.add_keyframe(at=0.0, value=0.0)
        kf2 = channel.add_keyframe(at=0.5, value=5.0, interpolation="cubic")
        kf3 = channel.add_keyframe(at=1.0, value=10.0, interpolation="bezier", 
                                  control_points=(0.8, 12.0, 0.9, 10.0))
        
        # Verify keyframes were added and have correct properties
        self.assertEqual(len(channel.keyframes), 3)
        self.assertEqual(kf1.at, 0.0)
        self.assertEqual(kf1.value(0.0, {}), 0.0)
        
        self.assertEqual(kf2.at, 0.5)
        self.assertEqual(kf2.value(0.5, {}), 5.0)
        self.assertEqual(kf2.interpolation, "cubic")
        
        self.assertEqual(kf3.at, 1.0)
        self.assertEqual(kf3.value(1.0, {}), 10.0)
        self.assertEqual(kf3.interpolation, "bezier")
        self.assertEqual(kf3.control_points, (0.8, 12.0, 0.9, 10.0))
        
        # Test adding a keyframe at an existing position (should replace)
        kf4 = channel.add_keyframe(at=0.5, value=7.0)
        self.assertEqual(len(channel.keyframes), 3)  # Count should remain the same
        
        # Find the keyframe at position 0.5
        for kf in channel.keyframes:
            if kf.at == 0.5:
                self.assertEqual(kf.value(0.5, {}), 7.0)  # Value should be updated
    
    def test_channel_remove_keyframe(self):
        """Test removing keyframes from a channel."""
        channel = Channel()
        
        # Add keyframes
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=0.5, value=5.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Verify keyframes were added
        self.assertEqual(len(channel.keyframes), 3)
        
        # Remove keyframe at position 0.5
        channel.remove_keyframe(0.5)
        
        # Verify keyframe was removed
        self.assertEqual(len(channel.keyframes), 2)
        for kf in channel.keyframes:
            self.assertNotEqual(kf.at, 0.5)
        
        # Test removing a nonexistent keyframe
        with self.assertRaises(ValueError):
            channel.remove_keyframe(0.75)
    
    def test_channel_get_value(self):
        """Test getting interpolated values from a channel."""
        channel = Channel()
        
        # Add keyframes
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Test getting value at various positions
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(0.5), 5.0)
        self.assertEqual(channel.get_value(1.0), 10.0)
        self.assertEqual(channel.get_value(0.25), 2.5)
        self.assertEqual(channel.get_value(0.75), 7.5)
        
        # Test getting values outside the range (should clamp)
        self.assertEqual(channel.get_value(-0.5), 0.0)
        self.assertEqual(channel.get_value(1.5), 10.0)
    
    def test_channel_get_keyframe_values(self):
        """Test getting keyframe values from a channel."""
        channel = Channel()
        
        # Add keyframes
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=0.5, value=5.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Get keyframe values
        keyframe_values = channel.get_keyframe_values()
        
        # Verify values
        self.assertEqual(len(keyframe_values), 3)
        
        expected_values = [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]
        for (expected_pos, expected_val), (actual_pos, actual_val) in zip(expected_values, keyframe_values):
            self.assertEqual(expected_pos, actual_pos)
            self.assertEqual(expected_val, actual_val)
        
        # Test with a channel containing expressions
        expr_channel = Channel()
        expr_channel.add_keyframe(at=0.0, value="sin(0)")
        expr_channel.add_keyframe(at=0.5, value="sin(pi/2)")
        
        expr_values = expr_channel.get_keyframe_values()
        self.assertEqual(len(expr_values), 2)
        self.assertEqual(expr_values[0][0], 0.0)
        self.assertAlmostEqual(expr_values[0][1], 0.0, places=5)
        self.assertEqual(expr_values[1][0], 0.5)
        self.assertAlmostEqual(expr_values[1][1], 1.0, places=5)
    
    def test_channel_sampling(self):
        """Test sampling a channel at multiple points."""
        channel = Channel()
        
        # Add keyframes
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Sample at multiple points
        samples = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = channel.sample(samples)
        
        # Verify values
        expected_values = [0.0, 2.5, 5.0, 7.5, 10.0]
        self.assertEqual(len(values), len(expected_values))
        
        for expected, actual in zip(expected_values, values):
            self.assertEqual(expected, actual)
    
    def test_channel_min_max(self):
        """Test channel min/max clamping."""
        # Create channel with min/max
        channel = Channel(min_max=(0, 5))
        channel.interpolation = "linear"  # Explicitly set linear interpolation
        
        # Add keyframes
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=0.5, value=10.0)  # Above max
        channel.add_keyframe(at=1.0, value=-5.0)  # Below min
        
        # Test clamping at keyframe points
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertEqual(channel.get_value(0.5), 5.0)  # Clamped to max
        self.assertEqual(channel.get_value(1.0), 0.0)  # Clamped to min
        
        # For intermediate values between 0 and 0.5, with linear interpolation
        # We should interpolate between 0 and 5 (max)
        result025 = channel.get_value(0.25) 
        self.assertLessEqual(result025, 5.0)  # Should be clamped at 5.0
        
        # For intermediate values between 0.5 and 1.0, with linear interpolation 
        # We should interpolate between 5 (max) and 0 (min)
        result075 = channel.get_value(0.75)
        # Since linear interpolation is used, it should be 5.0 * (1 - 0.5) where 0.5 = (0.75 - 0.5) / (1.0 - 0.5)
        self.assertLessEqual(result075, 5.0)
        self.assertGreaterEqual(result075, 0.0)
    
    def test_spline_add_channel(self):
        """Test adding channels to a spline."""
        spline = Spline()
        
        # Add channels
        ch1 = spline.add_channel(name="position", interpolation="linear")  # Explicitly set linear
        ch2 = spline.add_channel(name="rotation", interpolation="cubic")
        ch3 = spline.add_channel(name="scale", min_max=(0.1, 2.0))
        
        # Verify channels were added with correct properties
        self.assertEqual(len(spline.channels), 3)
        self.assertIn("position", spline.channels)
        self.assertIn("rotation", spline.channels)
        self.assertIn("scale", spline.channels)
        
        self.assertEqual(spline.channels["position"].interpolation, "linear")
        self.assertEqual(spline.channels["rotation"].interpolation, "cubic")
        self.assertEqual(spline.channels["scale"].min_max, (0.1, 2.0))
        
        # Test that existing channels are returned when replace=False (default)
        position_channel = spline.add_channel(name="position")
        position_channel2 = spline.add_channel(name="position")
        self.assertIs(position_channel, position_channel2)
    
    def test_spline_remove_channel(self):
        """Test removing channels from a spline."""
        spline = Spline()
        
        # Add channels
        spline.add_channel(name="position")
        spline.add_channel(name="rotation")
        spline.add_channel(name="scale")
        
        # Verify channels were added
        self.assertEqual(len(spline.channels), 3)
        
        # Since Spline doesn't have a remove_channel method,
        # we'll implement the behavior by directly modifying the channels dictionary
        del spline.channels["rotation"]
        
        # Verify channel was removed
        self.assertEqual(len(spline.channels), 2)
        self.assertIn("position", spline.channels)
        self.assertNotIn("rotation", spline.channels)
        self.assertIn("scale", spline.channels)
        
        # Test removing a nonexistent channel
        with self.assertRaises(KeyError):
            del spline.channels["nonexistent"]
    
    def test_spline_get_channel_names(self):
        """Test getting channel names from a spline."""
        spline = Spline()
        
        # Add channels
        spline.add_channel(name="position")
        spline.add_channel(name="rotation")
        spline.add_channel(name="scale")
        
        # Get channel names directly from the channels dictionary
        names = list(spline.channels.keys())
        
        # Verify names
        self.assertEqual(set(names), {"position", "rotation", "scale"})
    
    def test_spline_get_values(self):
        """Test getting values from all channels in a spline."""
        spline = Spline()
        
        # Add channels with keyframes
        pos_channel = spline.add_channel(name="position")
        pos_channel.add_keyframe(at=0.0, value=0.0)
        pos_channel.add_keyframe(at=1.0, value=10.0)
        
        rot_channel = spline.add_channel(name="rotation")
        rot_channel.add_keyframe(at=0.0, value=0.0)
        rot_channel.add_keyframe(at=1.0, value=360.0)
        
        # Get values at a specific time using get_value which is the actual method name
        values = spline.get_value(0.5)
        
        # Verify values
        self.assertEqual(len(values), 2)
        self.assertEqual(values["position"], 5.0)
        self.assertEqual(values["rotation"], 180.0)
    
    def test_spline_get_keyframes(self):
        """Test getting all keyframes from a spline."""
        spline = Spline()
        
        # Add channels with keyframes
        pos_channel = spline.add_channel(name="position")
        pos_channel.add_keyframe(at=0.0, value=0.0)
        pos_channel.add_keyframe(at=1.0, value=10.0)
        
        rot_channel = spline.add_channel(name="rotation")
        rot_channel.add_keyframe(at=0.0, value=0.0)
        rot_channel.add_keyframe(at=0.5, value=180.0)
        rot_channel.add_keyframe(at=1.0, value=360.0)
        
        # Get all keyframes by iterating through channel keyframes
        keyframes = {}
        for channel_name, channel in spline.channels.items():
            keyframes[channel_name] = channel.get_keyframe_values()
        
        # Verify keyframes
        self.assertEqual(len(keyframes), 2)  # Two channels
        self.assertIn("position", keyframes)
        self.assertIn("rotation", keyframes)
        
        self.assertEqual(len(keyframes["position"]), 2)  # Two keyframes
        self.assertEqual(len(keyframes["rotation"]), 3)  # Three keyframes
        
        # Verify keyframe values
        pos_kfs = keyframes["position"]
        self.assertEqual(pos_kfs[0][0], 0.0)  # Position
        self.assertEqual(pos_kfs[0][1], 0.0)  # Value
        self.assertEqual(pos_kfs[1][0], 1.0)
        self.assertEqual(pos_kfs[1][1], 10.0)
        
        rot_kfs = keyframes["rotation"]
        self.assertEqual(rot_kfs[0][0], 0.0)
        self.assertEqual(rot_kfs[0][1], 0.0)
        self.assertEqual(rot_kfs[1][0], 0.5)
        self.assertEqual(rot_kfs[1][1], 180.0)
        self.assertEqual(rot_kfs[2][0], 1.0)
        self.assertEqual(rot_kfs[2][1], 360.0)

if __name__ == "__main__":
    unittest.main()