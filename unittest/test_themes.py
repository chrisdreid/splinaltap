#!/usr/bin/env python3
"""
Tests for theme functionality in SplinalTap visualization.
"""

import os
import unittest
from unittest.mock import patch
import sys

# Add the parent directory to sys.path to allow importing from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import theme_examples module
from splinaltap.theme_examples import create_complex_solver

# Import the solver class
from splinaltap.solver import KeyframeSolver

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib is not available")
class TestThemes(unittest.TestCase):
    """Test case for theme functionality."""
    
    def setUp(self):
        """Set up a complex solver for testing."""
        self.solver = create_complex_solver()
        
        # Create test output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Temporary files that will be created during tests
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files."""
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_dark_theme(self):
        """Test the dark theme (default)."""
        plot_path = os.path.join(self.output_dir, "test_dark.png")
        self.temp_files.append(plot_path)
        
        # Generate and save plot with dark theme
        self.solver.save_plot(plot_path, theme="dark")
        
        # Verify file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_medium_theme(self):
        """Test the medium theme."""
        plot_path = os.path.join(self.output_dir, "test_medium.png")
        self.temp_files.append(plot_path)
        
        # Generate and save plot with medium theme
        self.solver.save_plot(plot_path, theme="medium")
        
        # Verify file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_light_theme(self):
        """Test the light theme."""
        plot_path = os.path.join(self.output_dir, "test_light.png")
        self.temp_files.append(plot_path)
        
        # Generate and save plot with light theme
        self.solver.save_plot(plot_path, theme="light")
        
        # Verify file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_default_theme_is_dark(self):
        """Test that the default theme is dark."""
        plot_path = os.path.join(self.output_dir, "test_default.png")
        self.temp_files.append(plot_path)
        
        # Save plot with default theme
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.style.use') as mock_style_use:
                self.solver.save_plot(plot_path)
                
                # Verify that dark theme was used
                mock_style_use.assert_called_with('dark_background')
    
    def test_overlay_option(self):
        """Test overlay option."""
        # Test overlay=True
        overlay_path = os.path.join(self.output_dir, "test_overlay_true.png")
        self.temp_files.append(overlay_path)
        self.solver.save_plot(overlay_path, overlay=True)
        self.assertTrue(os.path.exists(overlay_path))
        
        # Test overlay=False
        separate_path = os.path.join(self.output_dir, "test_overlay_false.png")
        self.temp_files.append(separate_path)
        self.solver.save_plot(separate_path, overlay=False)
        self.assertTrue(os.path.exists(separate_path))
    
    def test_filter_channels(self):
        """Test filter_channels option."""
        filter_path = os.path.join(self.output_dir, "test_filter.png")
        self.temp_files.append(filter_path)
        
        # Filter for only position.x and expressions.sine
        filter_channels = {
            "position": ["x"],
            "expressions": ["sine"]
        }
        
        self.solver.save_plot(filter_path, filter_channels=filter_channels)
        self.assertTrue(os.path.exists(filter_path))
    
    def test_single_spline_plot(self):
        """Test plotting a single spline."""
        spline_path = os.path.join(self.output_dir, "test_spline.png")
        self.temp_files.append(spline_path)
        
        # Get expressions spline and plot it
        expressions = self.solver.get_spline("expressions")
        expressions.save_plot(spline_path, title="Expressions")
        self.assertTrue(os.path.exists(spline_path))

if __name__ == "__main__":
    unittest.main()