"""
Theme examples for SplinalTap.

This module provides example solvers for testing and demonstrations.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union, Any

from .solver import SplineSolver

def create_complex_solver() -> SplineSolver:
    """Create a complex solver with multiple splines for demonstration and testing.
    
    Returns:
        A SplineSolver with various splines and keyframes
    """
    # Create a new solver
    solver = SplineSolver(name="Complex Animation")
    
    # Create a position spline group with x, y, z channels
    position = solver.create_spline("position")
    # Add default keyframes to the "value" channel to prevent errors
    value_channel = position.get_channel("value")
    value_channel.add_keyframe(at=0.0, value=0.0)
    value_channel.add_keyframe(at=1.0, value=1.0)
    x = position.add_channel("x")
    x.add_keyframe(at=0.0, value=0.0)
    x.add_keyframe(at=0.25, value=25.0)
    x.add_keyframe(at=0.5, value=50.0)
    x.add_keyframe(at=0.75, value=75.0)
    x.add_keyframe(at=1.0, value=100.0)
    
    # Make x channel published globally
    solver.set_publish("position.x", ["*"])
    
    y = position.add_channel("y")
    y.add_keyframe(at=0.0, value=0.0)
    y.add_keyframe(at=0.25, value=25.0)
    y.add_keyframe(at=0.5, value=50.0)
    y.add_keyframe(at=0.75, value=75.0)
    y.add_keyframe(at=1.0, value=100.0)
    
    # Add spline-level publish for y
    y.publish = ["expressions.*"]
    
    # Also add solver-level publish
    solver.set_publish("position.y", ["expressions.sine"])
    
    z = position.add_channel("z")
    z.add_keyframe(at=0.0, value=0.0)
    z.add_keyframe(at=0.5, value=50.0, interpolation="linear")
    z.add_keyframe(at=1.0, value=100.0)
    
    # Create a rotation spline group
    rotation = solver.create_spline("rotation")
    # Add default keyframes to the "value" channel to prevent errors
    value_channel = rotation.get_channel("value")
    value_channel.add_keyframe(at=0.0, value=0.0)
    value_channel.add_keyframe(at=1.0, value=1.0)
    
    angle = rotation.add_channel("angle")
    angle.add_keyframe(at=0.0, value=0.0)
    angle.add_keyframe(at=0.5, value=180.0)
    angle.add_keyframe(at=1.0, value=360.0)
    
    # Create an expressions spline group
    expressions = solver.create_spline("expressions")
    # Add default keyframes to the "value" channel to prevent errors
    value_channel = expressions.get_channel("value")
    value_channel.add_keyframe(at=0.0, value=0.0)
    value_channel.add_keyframe(at=1.0, value=1.0)
    
    # Add a sine expression
    sine = expressions.add_channel("sine")
    sine.add_keyframe(at=0.0, value="sin(position.x / 50 * 3.14159)")
    sine.add_keyframe(at=1.0, value="sin(position.x / 50 * 3.14159)")
    
    # Add a random expression
    random_ch = expressions.add_channel("random")
    random_ch.add_keyframe(at=0.0, value="rand() * 10")
    random_ch.add_keyframe(at=1.0, value="rand() * 10")
    
    # Add a dependent expression
    dependent = expressions.add_channel("dependent")
    dependent.add_keyframe(at=0.0, value="0.0")
    dependent.add_keyframe(at=0.5, value="position.x * expressions.sine")
    dependent.add_keyframe(at=1.0, value="position.x * expressions.sine * 2")
    
    # Print solver state
    print("Solver state after loading:")
    print(f"Publish rules: {solver.publish}")
    
    # Print channel publish rules
    channel_publish = {}
    for group_name, group in solver.spline_groups.items():
        for spline_name, spline in group.splines.items():
            if hasattr(spline, 'publish') and spline.publish:
                channel_publish[f"{group_name}.{spline_name}"] = spline.publish
    print(f"Spline publish rules: {channel_publish}")
    
    # Print a sample value
    print(f"Result at t=0.5: {solver.solve(0.5)}")
    
    return solver