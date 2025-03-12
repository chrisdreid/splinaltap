#!/usr/bin/env python3
"""Tests for file I/O operations with SplinalTap."""

import unittest
import os
import json
import tempfile
from splinaltap import KeyframeSolver
from splinaltap.backends import BackendManager

class TestFileIO(unittest.TestCase):
    """Test file I/O operations for the KeyframeSolver."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
        
        # Get the path to the test input file
        self.input_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'input.json'
        )
        
        # Create a temporary directory for output files
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_load_input_file(self):
        """Test loading a solver from a JSON file."""
        # Verify the input file exists
        self.assertTrue(os.path.exists(self.input_file), 
                       f"Test input file not found: {self.input_file}")
        
        # Load the solver
        solver = KeyframeSolver.load(self.input_file)
        
        # Verify basic solver properties
        self.assertEqual(solver.name, "TestScene")
        self.assertEqual(solver.get_metadata("author"), "SplinalTap Tests")
        self.assertEqual(solver.get_metadata("description"), "Test JSON file for unit tests")
        self.assertEqual(solver.range, (0, 1))
        
        # Verify variables
        self.assertAlmostEqual(solver.variables["pi"], 3.14159)
        self.assertEqual(solver.variables["amplitude"], 10)
        
        # Verify splines
        self.assertEqual(len(solver.splines), 3)
        self.assertIn("position", solver.splines)
        self.assertIn("rotation", solver.splines)
        self.assertIn("expressions", solver.splines)
        
        # Verify channels
        position_spline = solver.splines["position"]
        self.assertEqual(len(position_spline.channels), 3)
        self.assertIn("x", position_spline.channels)
        self.assertIn("y", position_spline.channels)
        self.assertIn("z", position_spline.channels)
        
        # Verify interpolation methods
        self.assertEqual(position_spline.channels["x"].interpolation, "linear")
        self.assertEqual(position_spline.channels["y"].interpolation, "cubic")
        self.assertEqual(position_spline.channels["z"].interpolation, "step")
        
        # Verify min/max values
        self.assertEqual(position_spline.channels["x"].min_max, (0, 100))
        
        # Verify keyframes
        x_keyframes = position_spline.channels["x"].keyframes
        self.assertEqual(len(x_keyframes), 3)
        
        # Verify expression channel
        expressions_spline = solver.splines["expressions"]
        self.assertIn("sine", expressions_spline.channels)
        self.assertIn("random", expressions_spline.channels)
        
        # Test evaluation of sine channel at key points
        sine_channel = expressions_spline.channels["sine"]
        self.assertAlmostEqual(sine_channel.get_value(0.0), 0.0, places=5)
        self.assertAlmostEqual(sine_channel.get_value(0.5), 1.0, places=5)
        self.assertAlmostEqual(sine_channel.get_value(1.0), 0.0, places=5)
    
    def test_save_and_reload(self):
        """Test saving a solver to a file and loading it back."""
        # Load the original solver
        original_solver = KeyframeSolver.load(self.input_file)
        
        # Create a temporary output file
        output_file = os.path.join(self.temp_dir.name, 'output.json')
        
        # Save the solver to the output file
        original_solver.save(output_file)
        
        # Verify the output file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Load the saved solver
        loaded_solver = KeyframeSolver.load(output_file)
        
        # Verify that the loaded solver matches the original
        self.assertEqual(loaded_solver.name, original_solver.name)
        self.assertEqual(loaded_solver.get_metadata("author"), original_solver.get_metadata("author"))
        self.assertEqual(loaded_solver.get_metadata("description"), original_solver.get_metadata("description"))
        self.assertEqual(loaded_solver.range, original_solver.range)
        
        # Verify variables
        for var_name, var_value in original_solver.variables.items():
            self.assertIn(var_name, loaded_solver.variables)
            if isinstance(var_value, float):
                self.assertAlmostEqual(loaded_solver.variables[var_name], var_value)
            else:
                self.assertEqual(loaded_solver.variables[var_name], var_value)
        
        # Verify splines
        self.assertEqual(len(loaded_solver.splines), len(original_solver.splines))
        for spline_name in original_solver.splines:
            self.assertIn(spline_name, loaded_solver.splines)
            
            # Verify channels
            original_spline = original_solver.splines[spline_name]
            loaded_spline = loaded_solver.splines[spline_name]
            
            self.assertEqual(len(loaded_spline.channels), len(original_spline.channels))
            for channel_name in original_spline.channels:
                self.assertIn(channel_name, loaded_spline.channels)
                
                # Verify channel properties
                original_channel = original_spline.channels[channel_name]
                loaded_channel = loaded_spline.channels[channel_name]
                
                self.assertEqual(loaded_channel.interpolation, original_channel.interpolation)
                if original_channel.min_max is not None:
                    self.assertEqual(loaded_channel.min_max, original_channel.min_max)
                
                # Verify keyframes
                self.assertEqual(len(loaded_channel.keyframes), len(original_channel.keyframes))
                
                # Sample both channels at various points to verify they produce the same values
                sample_points = [0.0, 0.25, 0.5, 0.75, 1.0]
                for at in sample_points:
                    original_value = original_channel.get_value(at)
                    loaded_value = loaded_channel.get_value(at)
                    self.assertAlmostEqual(loaded_value, original_value, places=5,
                                          msg=f"Value mismatch at {at} for {spline_name}.{channel_name}")
    
    def test_file_format_conversion(self):
        """Test converting between file formats."""
        # Load the original solver
        original_solver = KeyframeSolver.load(self.input_file)
        
        # Test JSON to YAML conversion (if PyYAML is available)
        try:
            import yaml
            
            # Create a temporary YAML output file
            yaml_file = os.path.join(self.temp_dir.name, 'output.yaml')
            
            # Save as YAML
            original_solver.save(yaml_file, format='yaml')
            
            # Verify the YAML file exists
            self.assertTrue(os.path.exists(yaml_file))
            
            # Load the YAML file
            loaded_solver = KeyframeSolver.load(yaml_file)
            
            # Verify that the loaded solver matches the original
            self.assertEqual(loaded_solver.name, original_solver.name)
            
            # Test converting back to JSON
            json_file = os.path.join(self.temp_dir.name, 'converted.json')
            loaded_solver.save(json_file, format='json')
            
            # Verify the JSON file exists
            self.assertTrue(os.path.exists(json_file))
            
        except ImportError:
            self.skipTest("PyYAML not available, skipping YAML conversion test")
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        # Test loading a nonexistent file
        nonexistent_file = os.path.join(self.temp_dir.name, 'nonexistent.json')
        with self.assertRaises(FileNotFoundError):
            KeyframeSolver.load(nonexistent_file)
        
        # Test loading a file with invalid JSON
        invalid_json_file = os.path.join(self.temp_dir.name, 'invalid.json')
        with open(invalid_json_file, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises(json.JSONDecodeError):
            KeyframeSolver.load(invalid_json_file)
        
        # Test loading a valid JSON file with incorrect structure
        invalid_structure_file = os.path.join(self.temp_dir.name, 'invalid_structure.json')
        with open(invalid_structure_file, 'w') as f:
            json.dump({"not_a_solver": True}, f)
        
        with self.assertRaises(KeyError):
            KeyframeSolver.load(invalid_structure_file)

if __name__ == "__main__":
    unittest.main()