"""
splinaltap - Keyframe interpolation and expression evaluation that goes to eleven!
"""

from .spline import Spline, Knot
from .spline import SplineGroup
from .solver import SplineSolver, KeyframeSolver
from .expression import ExpressionEvaluator
from .visualization import plot_interpolation_comparison, plot_single_interpolation

# For backward compatibility
Channel = Spline
Keyframe = Knot

__version__ = "0.8.0"
__all__ = [
    "SplineSolver",
    "KeyframeSolver",  # For backward compatibility
    "SplineGroup",
    "Spline", 
    "Knot",
    "Channel",  # For backward compatibility 
    "Keyframe",  # For backward compatibility
    "ExpressionEvaluator",
    "plot_interpolation_comparison", 
    "plot_single_interpolation",
]