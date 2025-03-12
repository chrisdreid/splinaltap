"""
Channel class for SplinalTap interpolation.

A Channel is a component of a Spline, representing a single animatable property
like the X coordinate of a position or the red component of a color.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .backends import BackendManager, get_math_functions
from .methods import (
    nearest_neighbor,
    linear_interpolate,
    polynomial_interpolate,
    quadratic_spline,
    hermite_interpolate,
    bezier_interpolate,
    gaussian_interpolate,
    pchip_interpolate,
    cubic_spline
)


class Keyframe:
    """A keyframe with position, value, interpolation method, and additional parameters."""
    
    def __init__(
        self, 
        at: float, 
        value: Union[float, str, Callable],
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ):
        """Initialize a keyframe.
        
        Args:
            at: The position of this keyframe (0-1 normalized)
            value: The value at this position (number, expression, or callable)
            interpolation: Optional interpolation method for this keyframe
            control_points: Optional control points for bezier interpolation
            derivative: Optional derivative value for hermite interpolation
        """
        self.at = at
        self.value = value
        self.interpolation = interpolation
        self.control_points = control_points
        self.derivative = derivative
        
    def __repr__(self) -> str:
        return f"Keyframe(at={self.at}, value={self.value}, interpolation={self.interpolation})"


class Channel:
    """A channel representing a single animatable property within a Spline."""
    
    def __init__(
        self, 
        interpolation: str = "cubic",
        min_max: Optional[Tuple[float, float]] = None,
        variables: Optional[Dict[str, Any]] = None
    ):
        """Initialize a channel.
        
        Args:
            interpolation: Default interpolation method for this channel
            min_max: Optional min/max range constraints for this channel's values
            variables: Optional variables to be used in expressions
        """
        self.interpolation = interpolation
        self.min_max = min_max
        self.keyframes: List[Keyframe] = []
        self.variables = variables or {}
        self._expression_evaluator = None
        
    def add_keyframe(
        self, 
        at: float, 
        value: Union[float, str], 
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ) -> None:
        """Add a keyframe to this channel.
        
        Args:
            at: The position of this keyframe (0-1 normalized)
            value: The value at this position (number or expression)
            interpolation: Optional interpolation method override for this keyframe
            control_points: Optional control points for bezier interpolation
            derivative: Optional derivative value for hermite interpolation
        """
        # Validate position range
        if not 0 <= at <= 1:
            raise ValueError(f"Keyframe position '@' must be between 0 and 1, got {at}")
            
        # Convert value to callable if it's an expression
        if isinstance(value, str):
            # Create expression evaluator if needed
            if self._expression_evaluator is None:
                from .expression import ExpressionEvaluator
                self._expression_evaluator = ExpressionEvaluator(self.variables)
            
            # Parse the expression
            value_callable = self._expression_evaluator.parse_expression(value)
        elif isinstance(value, (int, float)):
            # For constant values, create a simple callable
            value_callable = lambda t, channels={}: float(value)
        else:
            raise TypeError(f"Keyframe value must be a number or string expression, got {type(value).__name__}")
        
        # Create and add the keyframe
        keyframe = Keyframe(at, value_callable, interpolation, control_points, derivative)
        
        # Add to sorted position
        if not self.keyframes:
            self.keyframes.append(keyframe)
        else:
            # Find insertion position
            for i, kf in enumerate(self.keyframes):
                if at < kf.at:
                    self.keyframes.insert(i, keyframe)
                    break
                elif at == kf.at:
                    # Replace existing keyframe at this position
                    self.keyframes[i] = keyframe
                    break
            else:
                # Append at the end if position is greater than all existing keyframes
                self.keyframes.append(keyframe)
                
    def get_value(self, at: float, channels: Dict[str, float] = None) -> float:
        """Get the interpolated value at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            channels: Optional channel values to use in expressions
            
        Returns:
            The interpolated value at the specified position
        """
        if not self.keyframes:
            raise ValueError("Cannot get value: no keyframes defined")
            
        channels = channels or {}
        
        # If position is at or outside the range of keyframes, return the boundary keyframe value
        if at <= self.keyframes[0].at:
            return self.keyframes[0].value(at, channels)
        if at >= self.keyframes[-1].at:
            return self.keyframes[-1].value(at, channels)
            
        # Find the bracketing keyframes
        left_kf = None
        right_kf = None
        
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].at <= at <= self.keyframes[i + 1].at:
                left_kf = self.keyframes[i]
                right_kf = self.keyframes[i + 1]
                break
                
        if left_kf is None or right_kf is None:
            raise ValueError(f"Could not find bracketing keyframes for position {at}")
            
        # If the right keyframe has a specific interpolation method, use that
        # Otherwise use the channel's default method
        method = right_kf.interpolation or self.interpolation
        
        # Call the appropriate interpolation method
        return self._interpolate(method, at, left_kf, right_kf, channels)
        
    def _interpolate(
        self, 
        method: str, 
        at: float, 
        left_kf: Keyframe, 
        right_kf: Keyframe,
        channels: Dict[str, float]
    ) -> float:
        """Interpolate between two keyframes using the specified method.
        
        Args:
            method: The interpolation method to use
            at: The position to evaluate
            left_kf: The left bracketing keyframe
            right_kf: The right bracketing keyframe
            channels: Channel values to use in expressions
            
        Returns:
            The interpolated value
        """
        # Normalize position between the keyframes
        t_range = right_kf.at - left_kf.at
        if t_range <= 0:
            return left_kf.value(at, channels)
            
        t_norm = (at - left_kf.at) / t_range
        
        # Get keyframe values and convert to float if needed
        left_val = left_kf.value(left_kf.at, channels)
        right_val = right_kf.value(right_kf.at, channels)
        
        # Convert numpy arrays to Python float
        if hasattr(left_val, 'item') or hasattr(left_val, 'tolist'):
            left_val = float(left_val)
        if hasattr(right_val, 'item') or hasattr(right_val, 'tolist'):
            right_val = float(right_val)
        
        # Handle different interpolation methods
        if method == "nearest":
            return left_val if t_norm < 0.5 else right_val
            
        elif method == "linear":
            return left_val * (1 - t_norm) + right_val * t_norm
            
        elif method == "cubic":
            # For cubic interpolation, we need more keyframes ideally
            # This is a simplified implementation
            # Get keyframes for context
            kfs = self.keyframes
            idx = kfs.index(left_kf)
            
            # Get derivative approximations if not specified
            p0 = left_val
            p1 = right_val
            
            m0 = left_kf.derivative if left_kf.derivative is not None else self._estimate_derivative(idx) 
            m1 = right_kf.derivative if right_kf.derivative is not None else self._estimate_derivative(idx + 1)
            
            # Hermite basis functions
            h00 = 2*t_norm**3 - 3*t_norm**2 + 1
            h10 = t_norm**3 - 2*t_norm**2 + t_norm
            h01 = -2*t_norm**3 + 3*t_norm**2
            h11 = t_norm**3 - t_norm**2
            
            # Scale derivatives by the time range
            m0 *= t_range
            m1 *= t_range
            
            return h00*p0 + h10*m0 + h01*p1 + h11*m1
            
        elif method == "hermite":
            # Hermite interpolation
            p0 = left_val
            p1 = right_val
            
            m0 = left_kf.derivative if left_kf.derivative is not None else 0.0
            m1 = right_kf.derivative if right_kf.derivative is not None else 0.0
            
            # Hermite basis functions
            h00 = 2*t_norm**3 - 3*t_norm**2 + 1
            h10 = t_norm**3 - 2*t_norm**2 + t_norm
            h01 = -2*t_norm**3 + 3*t_norm**2
            h11 = t_norm**3 - t_norm**2
            
            # Scale derivatives by the time range
            m0 *= t_range
            m1 *= t_range
            
            return h00*p0 + h10*m0 + h01*p1 + h11*m1
            
        elif method == "bezier":
            # Get control points
            if right_kf.control_points and len(right_kf.control_points) >= 4:
                # Extract control points [p1_x, p1_y, p2_x, p2_y]
                cp = right_kf.control_points
                
                # Normalize control point x-coordinates to 0-1 range
                cp1_x = (cp[0] - left_kf.at) / t_range
                cp2_x = (cp[2] - left_kf.at) / t_range
                
                # De Casteljau algorithm for a single parametric value
                # This is simplified and could be optimized
                def de_casteljau(t):
                    # Start with the control points
                    p0 = (0.0, left_val)  # Start point
                    p1 = (cp1_x, cp[1])   # First control point
                    p2 = (cp2_x, cp[3])   # Second control point
                    p3 = (1.0, right_val) # End point
                    
                    # Interpolate between points
                    q0 = (p0[0]*(1-t) + p1[0]*t, p0[1]*(1-t) + p1[1]*t)
                    q1 = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
                    q2 = (p2[0]*(1-t) + p3[0]*t, p2[1]*(1-t) + p3[1]*t)
                    
                    # Second level of interpolation
                    r0 = (q0[0]*(1-t) + q1[0]*t, q0[1]*(1-t) + q1[1]*t)
                    r1 = (q1[0]*(1-t) + q2[0]*t, q1[1]*(1-t) + q2[1]*t)
                    
                    # Final interpolation gives the point on the curve
                    result = (r0[0]*(1-t) + r1[0]*t, r0[1]*(1-t) + r1[1]*t)
                    
                    return result[1]  # Return the y-coordinate
                
                return de_casteljau(t_norm)
            else:
                # Fallback to cubic interpolation if control points aren't available
                return self._interpolate("cubic", at, left_kf, right_kf, channels)
                
        # For other methods, we could add more specialized implementations
        # For now, default to cubic for anything else
        return self._interpolate("cubic", at, left_kf, right_kf, channels)
        
    def _estimate_derivative(self, idx: int) -> float:
        """Estimate the derivative at a keyframe position.
        
        Args:
            idx: The index of the keyframe in the keyframes list
            
        Returns:
            Estimated derivative value
        """
        kfs = self.keyframes
        
        # Handle boundary cases
        if idx <= 0:
            # At the start, use forward difference
            if len(kfs) > 1:
                p0 = kfs[0].value(kfs[0].at, {})
                p1 = kfs[1].value(kfs[1].at, {})
                t0 = kfs[0].at
                t1 = kfs[1].at
                return (p1 - p0) / (t1 - t0) if t1 > t0 else 0.0
            return 0.0
            
        elif idx >= len(kfs) - 1:
            # At the end, use backward difference
            if len(kfs) > 1:
                p0 = kfs[-2].value(kfs[-2].at, {})
                p1 = kfs[-1].value(kfs[-1].at, {})
                t0 = kfs[-2].at
                t1 = kfs[-1].at
                return (p1 - p0) / (t1 - t0) if t1 > t0 else 0.0
            return 0.0
            
        else:
            # In the middle, use central difference
            p0 = kfs[idx-1].value(kfs[idx-1].at, {})
            p1 = kfs[idx+1].value(kfs[idx+1].at, {})
            t0 = kfs[idx-1].at
            t1 = kfs[idx+1].at
            return (p1 - p0) / (t1 - t0) if t1 > t0 else 0.0
    
    def get_keyframe_values(self, channels: Dict[str, float] = None) -> List[Tuple[float, float]]:
        """Get all keyframe positions and values.
        
        Args:
            channels: Optional channel values to use in expressions
            
        Returns:
            List of (position, value) tuples
        """
        channels = channels or {}
        
        return [(kf.at, kf.value(kf.at, channels)) for kf in self.keyframes]