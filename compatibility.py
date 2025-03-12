"""
Compatibility module for SplinalTap.

This module provides compatibility classes and functions for backward compatibility
with older code that may still be using the old API.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .channel import Channel
from .spline import Spline
from .solver import KeyframeSolver

class KeyframeInterpolator:
    """
    Compatibility class that mimics the old KeyframeInterpolator API.
    
    This class wraps the new architecture (KeyframeSolver, Spline, Channel)
    to provide backward compatibility with older code that may still be using
    the old API.
    """
    
    def __init__(self, 
                interpolation: str = "cubic", 
                min_max: Optional[Tuple[float, float]] = None,
                variables: Optional[Dict[str, Any]] = None):
        """
        Initialize a KeyframeInterpolator.
        
        Args:
            interpolation: Default interpolation method
            min_max: Optional min/max range constraints
            variables: Optional variables for expressions
        """
        self.solver = KeyframeSolver()
        self.spline = self.solver.create_spline("main")
        self.channel = self.spline.add_channel("value", interpolation=interpolation, min_max=min_max)
        
        # Set variables if provided
        if variables:
            for name, value in variables.items():
                self.solver.set_variable(name, value)
        
        self.keyframes = []  # For compatibility with old API
    
    def set_keyframe(self, at: float, value: Union[float, str], 
                    method: Optional[str] = None,
                    derivative: Optional[float] = None,
                    control_points: Optional[List[float]] = None) -> None:
        """
        Add a keyframe to the interpolator.
        
        Args:
            at: The position of this keyframe (0-1 normalized)
            value: The value at this position
            method: Optional interpolation method
            derivative: Optional derivative for hermite interpolation
            control_points: Optional control points for bezier interpolation
        """
        self.channel.add_keyframe(at=at, value=value, interpolation=method,
                                derivative=derivative, control_points=control_points)
        
        # Update the keyframes list for compatibility
        self.keyframes = self.channel.keyframes
    
    def get_value(self, at: float, method: Optional[str] = None) -> float:
        """
        Get the interpolated value at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            method: Optional interpolation method override
            
        Returns:
            The interpolated value at the specified position
        """
        # If method is provided, temporarily override the channel's method
        original_method = self.channel.interpolation
        if method is not None:
            self.channel.interpolation = method
            
        try:
            result = self.channel.get_value(at)
        finally:
            # Restore the original method
            if method is not None:
                self.channel.interpolation = original_method
                
        return result
    
    def get_time_range(self) -> Tuple[float, float]:
        """
        Get the time range of this interpolator.
        
        Returns:
            Tuple of (min_time, max_time)
        """
        if not self.channel.keyframes:
            return (0.0, 0.0)
        
        return (self.channel.keyframes[0].at, self.channel.keyframes[-1].at)
    
    def sample_range(self, start: float, end: float, count: int, 
                    method: Optional[str] = None) -> List[float]:
        """
        Sample the interpolator over a range of values.
        
        Args:
            start: Start of the range
            end: End of the range
            count: Number of samples to take
            method: Optional interpolation method override
            
        Returns:
            List of interpolated values
        """
        # Generate evenly spaced positions
        step = (end - start) / (count - 1) if count > 1 else 0
        positions = [start + i * step for i in range(count)]
        
        # Evaluate at each position
        return [self.get_value(pos, method) for pos in positions]
    
    def sample_with_gpu(self, start: float, end: float, count: int,
                        method: Optional[str] = None) -> List[float]:
        """
        Sample the interpolator over a range of values using GPU if available.
        
        Args:
            start: Start of the range
            end: End of the range
            count: Number of samples to take
            method: Optional interpolation method override
            
        Returns:
            List of interpolated values
        """
        # Try to use a GPU backend if available
        from .backends import BackendManager
        original_backend = BackendManager.get_backend().name
        
        try:
            # Try to use CuPy or JAX backend
            if "cupy" in BackendManager.available_backends():
                BackendManager.set_backend("cupy")
            elif "jax" in BackendManager.available_backends():
                BackendManager.set_backend("jax")
            
            # Sample normally - the backend switch will use GPU if available
            result = self.sample_range(start, end, count, method)
        finally:
            # Restore original backend
            BackendManager.set_backend(original_backend)
            
        return result
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable for use in expressions.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.solver.set_variable(name, value)