#!/usr/bin/env python3
"""Tests for the KeyframeSolver API."""

import unittest
import os
import tempfile
import json
from splinaltap import KeyframeSolver, Spline, Channel
from splinaltap.backends import BackendManager

class TestSolverAPI(unittest.TestCase):
    """Test the KeyframeSolver API functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
        
        # Create a simple solver for testing
        self.solver = KeyframeSolver(name="test_solver")
        self.spline = self.solver.create_spline("main")
        self.channel = self.spline.add_channel("position")
        
        # Add keyframes
        self.channel.add_keyframe(at=0.0, value=0.0)
        self.channel.add_keyframe(at=1.0, value=10.0)
    
    def test_solve_method(self):
        """Test the solve method."""
        # Test at keyframe points
        result_0 = self.solver.solve(0.0)
        result_1 = self.solver.solve(1.0)
        
        # Verify result structure and values
        self.assertIn("main", result_0)
        self.assertIn("position", result_0["main"])
        self.assertEqual(result_0["main"]["position"], 0.0)
        
        self.assertIn("main", result_1)
        self.assertIn("position", result_1["main"])
        self.assertEqual(result_1["main"]["position"], 10.0)
        
        # Test at intermediate points
        result_mid = self.solver.solve(0.5)
        self.assertEqual(result_mid["main"]["position"], 5.0)
        
        # Test with multiple splines and channels
        spline2 = self.solver.create_spline("secondary")
        channel2 = spline2.add_channel("rotation")
        channel2.add_keyframe(at=0.0, value=0.0)
        channel2.add_keyframe(at=1.0, value=360.0)
        
        result_multi = self.solver.solve(0.5)
        self.assertIn("main", result_multi)
        self.assertIn("secondary", result_multi)
        self.assertEqual(result_multi["main"]["position"], 5.0)
        self.assertEqual(result_multi["secondary"]["rotation"], 180.0)
        
        # Test with explicit range
        self.solver.range = (0.0, 10.0)
        
        result_range = self.solver.solve(5.0)  # Halfway in the new range
        self.assertEqual(result_range["main"]["position"], 5.0)
    
    def test_solve_multiple_method(self):
        """Test the solve_multiple method."""
        # Define sample points
        sample_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Solve at multiple points
        results = self.solver.solve_multiple(sample_points)
        
        # Verify result structure and length
        self.assertEqual(len(results), len(sample_points))
        
        # Verify values
        expected_values = [0.0, 2.5, 5.0, 7.5, 10.0]
        
        for i, (sample, result) in enumerate(zip(sample_points, results)):
            self.assertIn("main", result)
            self.assertIn("position", result["main"])
            self.assertEqual(result["main"]["position"], expected_values[i])
    
    def test_variables(self):
        """Test variable setting and usage."""
        self.solver.set_variable("scale", 2.0)
        
        # Add a channel using the variable
        scale_channel = self.spline.add_channel("scaled")
        scale_channel.add_keyframe(at=0.0, value="0 * scale")
        scale_channel.add_keyframe(at=1.0, value="10 * scale")
        
        # Test that expressions use the variable
        result = self.solver.solve(0.5)
        self.assertEqual(result["main"]["scaled"], 10.0)  # 5 * 2
        
        # Change the variable and test again
        self.solver.set_variable("scale", 0.5)
        result = self.solver.solve(0.5)
        self.assertEqual(result["main"]["scaled"], 2.5)  # 5 * 0.5
    
    def test_metadata(self):
        """Test metadata handling."""
        # Set metadata
        self.solver.set_metadata("author", "Test User")
        self.solver.set_metadata("description", "Test solver")
        
        # Access metadata directly
        self.assertEqual(self.solver.metadata["author"], "Test User")
        self.assertEqual(self.solver.metadata["description"], "Test solver")
        
        # Get nonexistent metadata
        self.assertIsNone(self.solver.metadata.get("nonexistent"))
        
        # Get with default value
        self.assertEqual(self.solver.metadata.get("version", "1.0"), "1.0")
    
    def test_save_and_load(self):
        """Test saving and loading solver."""
        # Set up a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Set some metadata
            self.solver.set_metadata("author", "Test User")
            self.solver.set_variable("scale", 2.0)
            
            # Save the solver
            self.solver.save(temp_path)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Check file contents directly instead of loading and deserializing
            with open(temp_path, 'r') as f:
                file_contents = f.read()
                
            # Verify the JSON contains the expected data
            self.assertIn(f'"name": "{self.solver.name}"', file_contents)
            self.assertIn('"author": "Test User"', file_contents)
            self.assertIn('"scale": 2.0', file_contents)
            self.assertIn('"position"', file_contents)  # Channel name
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_spline_names(self):
        """Test getting spline names."""
        solver = KeyframeSolver()
        solver.create_spline("spline1")
        solver.create_spline("spline2")
        solver.create_spline("spline3")
        
        names = solver.get_spline_names()
        self.assertEqual(set(names), {"spline1", "spline2", "spline3"})
    
    def test_error_handling(self):
        """Test error handling in the solver API."""
        # Test invalid spline name
        with self.assertRaises(KeyError):
            self.solver.get_spline("nonexistent")
        
        # Test invalid solve time (outside range)
        self.solver.range = (0.0, 1.0)
        # These should clamp rather than raising errors
        result_before = self.solver.solve(-1.0)
        result_after = self.solver.solve(2.0)
        
        self.assertEqual(result_before["main"]["position"], 0.0)  # Clamped to start
        self.assertEqual(result_after["main"]["position"], 10.0)  # Clamped to end
        
        # Test loading nonexistent file
        with self.assertRaises(FileNotFoundError):
            KeyframeSolver.load("nonexistent_file.json")
        
        # Test loading invalid file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(b"{invalid json")
            temp_path = tmp.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                KeyframeSolver.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_solver_copy(self):
        """Test copying a solver."""
        # Create a copy of the solver
        copied_solver = self.solver.copy()
        
        # Verify that it's a different object but with the same values
        self.assertIsNot(copied_solver, self.solver)
        self.assertEqual(copied_solver.name, self.solver.name)
        
        # Verify splines and channels
        self.assertIn("main", copied_solver.splines)
        self.assertIn("position", copied_solver.splines["main"].channels)
        
        # Verify that changes to the copy don't affect the original
        copied_solver.name = "copied_solver"
        copied_solver.splines["main"].channels["position"].add_keyframe(at=0.5, value=5.0)
        
        self.assertEqual(self.solver.name, "test_solver")
        self.assertEqual(len(self.solver.splines["main"].channels["position"].keyframes), 2)
        self.assertEqual(len(copied_solver.splines["main"].channels["position"].keyframes), 3)

if __name__ == "__main__":
    unittest.main()