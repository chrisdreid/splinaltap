#!/usr/bin/env python3
"""
Test script for figure size parameters in SplinalTap.
This script tests that width and height parameters are applied correctly.
"""

from splinaltap.solver import KeyframeSolver
import matplotlib.pyplot as plt
import os

def test_figure_sizes():
    """Test that width and height parameters work correctly."""
    
    # Create a simple solver
    solver = KeyframeSolver(name="SizeTest")
    
    # Create a simple spline with one channel
    position = solver.create_spline("position")
    x = position.add_channel("x", interpolation="cubic")
    
    # Add keyframes
    x.add_keyframe(at=0.0, value=0.0)
    x.add_keyframe(at=0.25, value=5.0)
    x.add_keyframe(at=0.5, value=0.0)
    x.add_keyframe(at=0.75, value=-5.0)
    x.add_keyframe(at=1.0, value=0.0)
    
    # Create output directory if it doesn't exist
    if not os.path.exists('size_test_output'):
        os.makedirs('size_test_output')
    
    # Test different figure sizes
    
    # Default size
    default_path = 'size_test_output/default_size.png'
    solver.save_plot(default_path, samples=100, theme="dark")
    print(f"Saved default size plot to {default_path}")
    
    # Wide figure (20x6)
    wide_path = 'size_test_output/wide_figure.png'
    solver.save_plot(wide_path, samples=100, theme="dark", width=20, height=6)
    print(f"Saved wide figure (20x6) to {wide_path}")
    
    # Tall figure (8x12)
    tall_path = 'size_test_output/tall_figure.png'
    solver.save_plot(tall_path, samples=100, theme="dark", width=8, height=12)
    print(f"Saved tall figure (8x12) to {tall_path}")
    
    # Small figure (6x4)
    small_path = 'size_test_output/small_figure.png'
    solver.save_plot(small_path, samples=100, theme="dark", width=6, height=4)
    print(f"Saved small figure (6x4) to {small_path}")
    
    # Test spline plot with size parameters
    spline = solver.get_spline("position")
    
    # Default spline size
    spline_default_path = 'size_test_output/spline_default_size.png'
    spline.save_plot(spline_default_path, samples=100, theme="dark")
    print(f"Saved default spline size plot to {spline_default_path}")
    
    # Custom spline size
    spline_custom_path = 'size_test_output/spline_custom_size.png'
    spline.save_plot(spline_custom_path, samples=100, theme="dark", width=16, height=5)
    print(f"Saved custom spline size (16x5) plot to {spline_custom_path}")
    
    print("Test complete. Check the output files to confirm figure sizes are applied correctly.")

if __name__ == "__main__":
    test_figure_sizes()