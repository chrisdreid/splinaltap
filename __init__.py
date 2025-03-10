"""
splinaltap - Keyframe interpolation and expression evaluation that goes to eleven!
"""

from .interpolator import KeyframeInterpolator
from .visualization import plot_interpolation_comparison, plot_single_interpolation

__version__ = "0.1.0"
__all__ = [
    "KeyframeInterpolator", 
    "plot_interpolation_comparison", 
    "plot_single_interpolation"
]