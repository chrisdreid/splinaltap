"""
Unit tests for the plotting functionality in SplinalTap.
"""

import unittest
import sys
import os
import io
from unittest.mock import patch, MagicMock

# Add parent directory to Python path for proper import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from splinaltap import KeyframeSolver
from splinaltap.spline import Spline

# Try to import matplotlib for tests that require it
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestPlotting(unittest.TestCase):
    """Test plotting functionality in SplinalTap."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple solver for testing
        self.solver = KeyframeSolver(name="test_solver")
        self.spline = self.solver.create_spline("main")
        self.channel = self.spline.add_channel("value")
        
        # Add keyframes
        self.channel.add_keyframe(at=0.0, value=0.0)
        self.channel.add_keyframe(at=0.5, value=5.0)
        self.channel.add_keyframe(at=1.0, value=0.0)
    
    def test_matplotlib_requirement(self):
        """Test that plotting requires matplotlib."""
        # Use unittest mock to simulate matplotlib not being available
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
            # This should raise an ImportError
            with self.assertRaises(ImportError):
                self.spline.plot()
                
            with self.assertRaises(ImportError):
                self.solver.plot()
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_spline_get_plot(self):
        """Test Spline.get_plot method."""
        # Test that get_plot returns a matplotlib figure
        fig = self.spline.get_plot()
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
        
        # Test with custom parameters
        fig = self.spline.get_plot(samples=50, filter_channels=["value"], theme="dark", title="Test Plot")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_spline_plot(self):
        """Test Spline.plot method."""
        # Redirect stdout to capture output
        with patch('matplotlib.pyplot.show') as mock_show:
            # Test that plot calls plt.show()
            result = self.spline.plot()
            self.assertIsNone(result)  # plot should return None
            mock_show.assert_called_once()
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_solver_get_plot(self):
        """Test Solver.get_plot method."""
        # Test that get_plot returns a matplotlib figure
        fig = self.solver.get_plot()
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
        
        # Test with custom parameters
        fig = self.solver.get_plot(samples=50, filter_channels={"main": ["value"]}, theme="dark")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_solver_plot(self):
        """Test Solver.plot method."""
        # Redirect stdout to capture output
        with patch('matplotlib.pyplot.show') as mock_show:
            # Test that plot calls plt.show()
            result = self.solver.plot()
            self.assertIsNone(result)  # plot should return None
            mock_show.assert_called_once()
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_complex_plotting(self):
        """Test plotting with multiple splines and channels."""
        # Create a more complex solver with multiple splines and channels
        solver = KeyframeSolver(name="complex_solver")
        
        # Create position spline with x, y channels
        position = solver.create_spline("position")
        x = position.add_channel("x")
        y = position.add_channel("y")
        
        x.add_keyframe(at=0.0, value=0.0)
        x.add_keyframe(at=1.0, value=10.0)
        
        y.add_keyframe(at=0.0, value=0.0)
        y.add_keyframe(at=1.0, value=5.0)
        
        # Create rotation spline with angle channel
        rotation = solver.create_spline("rotation")
        angle = rotation.add_channel("angle")
        
        angle.add_keyframe(at=0.0, value=0.0)
        angle.add_keyframe(at=1.0, value=360.0)
        
        # Test that we can get a plot for this complex solver
        fig = solver.get_plot()
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
        
        # Test plotting with filter
        fig = solver.get_plot(filter_channels={"position": ["x"]})
        self.assertIsNotNone(fig)
        
        # Test get_plot works on individual splines
        fig = position.get_plot()
        self.assertIsNotNone(fig)
    
    @unittest.skipIf(not HAS_MATPLOTLIB, "Matplotlib not available")
    def test_plotting_with_expressions(self):
        """Test plotting with expression-based channels."""
        # Create a solver with expression-based channels
        solver = KeyframeSolver(name="expression_solver")
        
        # Create splines
        base = solver.create_spline("base")
        derived = solver.create_spline("derived")
        
        # Add channels
        value = base.add_channel("value")
        squared = derived.add_channel("squared")
        
        # Add keyframes
        value.add_keyframe(at=0.0, value=0.0)
        value.add_keyframe(at=1.0, value=5.0)
        
        # Set up publishing
        solver.set_publish("base.value", ["derived.squared"])
        
        # Add expression-based keyframes
        squared.add_keyframe(at=0.0, value="base.value * base.value")
        squared.add_keyframe(at=1.0, value="base.value * base.value")
        
        # Test that we can get a plot
        fig = solver.get_plot()
        self.assertIsNotNone(fig)
        
        # Test that the derived channel is plotted correctly
        fig = derived.get_plot()
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()