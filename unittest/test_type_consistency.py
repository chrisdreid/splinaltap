#!/usr/bin/env python3
"""Tests for type consistency across the SplinalTap API."""

import unittest
import sys
import os

from splinaltap import KeyframeSolver, Spline, Channel
from splinaltap.backends import BackendManager

class TestTypeConsistency(unittest.TestCase):
    """Test that the API returns consistent Python native types."""

    def test_channel_get_value_returns_float(self):
        """Test that Channel.get_value() always returns a Python float."""
        solver = KeyframeSolver()
        spline = solver.create_spline("test")
        channel = spline.add_channel("value")
        
        # Add various types of keyframes
        channel.add_keyframe(at=0.0, value=0)
        channel.add_keyframe(at=0.25, value="sin(t) + 1")
        channel.add_keyframe(at=0.5, value="t^2")
        channel.add_keyframe(at=0.75, value="rand()")
        channel.add_keyframe(at=1.0, value=10)
        
        # Test values at various positions
        positions = [i * 0.1 for i in range(11)]
        
        for backend_name in BackendManager.available_backends():
            with self.subTest(backend=backend_name):
                BackendManager.set_backend(backend_name)
                values = [channel.get_value(p) for p in positions]
                
                # Check all values are Python floats
                for val in values:
                    self.assertIsInstance(val, float)

    def test_solver_returns_consistent_types(self):
        """Test that KeyframeSolver.solve() returns consistent Python types."""
        solver = KeyframeSolver()
        spline = solver.create_spline("main")
        
        # Add channels with different expression types
        channel1 = spline.add_channel("simple")
        channel1.add_keyframe(at=0.0, value=0)
        channel1.add_keyframe(at=1.0, value=10)
        
        channel2 = spline.add_channel("expr")
        channel2.add_keyframe(at=0.0, value="sin(t*pi)")
        channel2.add_keyframe(at=1.0, value="cos(t*pi)")
        
        channel3 = spline.add_channel("random")
        channel3.add_keyframe(at=0.0, value="rand()")
        channel3.add_keyframe(at=1.0, value="randint(5)")
        
        # Test values with each backend
        for backend_name in BackendManager.available_backends():
            with self.subTest(backend=backend_name):
                BackendManager.set_backend(backend_name)
                
                result = solver.solve(0.5)
                
                # Check all values are Python floats
                for spline_name, channels in result.items():
                    for channel_name, value in channels.items():
                        self.assertIsInstance(value, float)

    def test_keyframe_values_returns_consistent_types(self):
        """Test that Channel.get_keyframe_values() returns consistent Python types."""
        channel = Channel()
        
        # Add keyframes with various value types
        channel.add_keyframe(at=0.0, value=0)
        channel.add_keyframe(at=0.33, value="sin(t)")
        channel.add_keyframe(at=0.66, value="rand()")
        channel.add_keyframe(at=1.0, value=10)
        
        for backend_name in BackendManager.available_backends():
            with self.subTest(backend=backend_name):
                BackendManager.set_backend(backend_name)
                
                keyframe_values = channel.get_keyframe_values()
                
                # Check all positions and values are Python floats
                for pos, val in keyframe_values:
                    self.assertIsInstance(pos, float)
                    self.assertIsInstance(val, float)

if __name__ == "__main__":
    unittest.main()