"""
splinaltap - Keyframe interpolation and expression evaluation that goes to eleven!
"""

from .channel import Channel, Keyframe
from .spline import Spline
from .solver import KeyframeSolver
from .expression import ExpressionEvaluator
from .visualization import plot_interpolation_comparison, plot_single_interpolation

__version__ = "0.8.0"
__all__ = [
    "KeyframeSolver",
    "Spline",
    "Channel", 
    "Keyframe",
    "ExpressionEvaluator",
    "plot_interpolation_comparison", 
    "plot_single_interpolation",
]