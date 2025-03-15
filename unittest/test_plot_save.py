#!/usr/bin/env python3
"""
Tests for plot saving functionality in SplinalTap.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Import the relevant classes
from splinaltap.solver import KeyframeSolver
from splinaltap.spline import Spline

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib is not available")
class TestPlotSave(unittest.TestCase):
    """Test case for plot saving functionality."""
    
    def setUp(self):
        """Set up a simple solver for testing."""
        self.solver = KeyframeSolver(name="TestSolver")
        
        # Create a position spline with x, y, z channels
        self.position = self.solver.create_spline("position")
        
        # Add channels with different interpolation methods
        self.x = self.position.add_channel("x")
        self.y = self.position.add_channel("y", interpolation="linear")
        
        # Add some keyframes
        self.x.add_keyframe(at=0.0, value=0.0)
        self.x.add_keyframe(at=0.5, value=5.0)
        self.x.add_keyframe(at=1.0, value=0.0)
        
        self.y.add_keyframe(at=0.0, value=0.0)
        self.y.add_keyframe(at=0.5, value=10.0)
        self.y.add_keyframe(at=1.0, value=0.0)
        
        # Temporary files for testing
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files."""
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)
    
    @patch('matplotlib.pyplot.savefig')
    def test_solver_get_plot_save(self, mock_savefig):
        """Test the solver's get_plot method with save_path parameter."""
        self.solver.get_plot(save_path="test_solver.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_solver.png")
    
    @patch('matplotlib.pyplot.savefig')
    def test_spline_get_plot_save(self, mock_savefig):
        """Test the spline's get_plot method with save_path parameter."""
        self.position.get_plot(save_path="test_spline.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_spline.png")
    
    @patch('matplotlib.pyplot.savefig')
    def test_solver_save_plot(self, mock_savefig):
        """Test the solver's save_plot method."""
        self.solver.save_plot("test_solver_save.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_solver_save.png")
    
    @patch('matplotlib.pyplot.savefig')
    def test_spline_save_plot(self, mock_savefig):
        """Test the spline's save_plot method."""
        self.position.save_plot("test_spline_save.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_spline_save.png")
    
    @patch('matplotlib.pyplot.show')
    def test_solver_show(self, mock_show):
        """Test the solver's show method."""
        self.solver.show()
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_spline_show(self, mock_show):
        """Test the spline's show method."""
        self.position.show()
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    def test_solver_plot_with_save(self, mock_savefig):
        """Test the solver's plot method with save_path parameter."""
        # Mock the show method to avoid displaying the plot
        with patch('matplotlib.pyplot.show'):
            self.solver.plot(save_path="test_solver_plot.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_solver_plot.png")
    
    @patch('matplotlib.pyplot.savefig')
    def test_spline_plot_with_save(self, mock_savefig):
        """Test the spline's plot method with save_path parameter."""
        # Mock the show method to avoid displaying the plot
        with patch('matplotlib.pyplot.show'):
            self.position.plot(save_path="test_spline_plot.png")
        mock_savefig.assert_called_once()
        # Add the filename to the temp files list for cleanup
        self.temp_files.append("test_spline_plot.png")
    
    def test_actual_file_creation(self):
        """Test that files are actually created (not just mock calls)."""
        # Turn off interactive mode to avoid showing plots
        plt.ioff()
        
        # Create and check solver file
        solver_file = "solver_actual.png"
        self.solver.save_plot(solver_file)
        self.assertTrue(os.path.exists(solver_file))
        self.temp_files.append(solver_file)
        
        # Create and check spline file
        spline_file = "spline_actual.png"
        self.position.save_plot(spline_file)
        self.assertTrue(os.path.exists(spline_file))
        self.temp_files.append(spline_file)
        
        # Turn interactive mode back on
        plt.ion()

if __name__ == "__main__":
    unittest.main()