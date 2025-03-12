"""
splinaltap - Keyframe interpolation and expression evaluation that goes to eleven!
"""

# Original classes (will be deprecated)
from .interpolator import KeyframeInterpolator
from .scene import Scene

# New architecture
from .channel import Channel, Keyframe
from .spline import Spline
from .solver import Solver
from .expression import ExpressionEvaluator

# Visualization (to be updated for new architecture)
from .visualization import plot_interpolation_comparison, plot_single_interpolation

__version__ = "0.2.0"
__all__ = [
    # New architecture
    "Solver",
    "Spline",
    "Channel", 
    "Keyframe",
    "ExpressionEvaluator",
    
    # Visualization
    "plot_interpolation_comparison", 
    "plot_single_interpolation",
    
    # Legacy classes (will be deprecated)
    "KeyframeInterpolator",
    "Scene"
]