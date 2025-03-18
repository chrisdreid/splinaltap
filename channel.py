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
            
        Raises:
            TypeError: If at is not a number
            TypeError: If value is not a number, string, or callable
            TypeError: If interpolation is not None or a string
            TypeError: If control_points is not None or a list of floats
            TypeError: If derivative is not None or a float
        """
        # Type check at
        if not isinstance(at, (int, float)):
            raise TypeError(f"'at' parameter must be a number, got {type(at).__name__}")
            
        # Type check value
        if not isinstance(value, (int, float, str)) and not callable(value):
            raise TypeError(f"Value must be a number, string, or callable, got {type(value).__name__}")
            
        # Type check interpolation
        if interpolation is not None and not isinstance(interpolation, str):
            raise TypeError(f"Interpolation must be a string or None, got {type(interpolation).__name__}")
            
        # Type check control_points
        if control_points is not None:
            if not isinstance(control_points, (list, tuple)):
                raise TypeError(f"Control points must be a list or tuple, got {type(control_points).__name__}")
            if not all(isinstance(point, (int, float)) for point in control_points):
                raise TypeError("All control points must be numbers (int or float)")
                
        # Type check derivative
        if derivative is not None and not isinstance(derivative, (int, float)):
            raise TypeError(f"Derivative must be a number or None, got {type(derivative).__name__}")
            
        self.at = at  # Keep the 'at' parameter name for backward compatibility
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
        variables: Optional[Dict[str, Any]] = None,
        publish: Optional[List[str]] = None
    ):
        """Initialize a channel.
        
        Args:
            interpolation: Default interpolation method for this channel
            min_max: Optional min/max range constraints for this channel's values
            variables: Optional variables to be used in expressions
            publish: Optional list of channel references to publish this channel's value to
            
        Raises:
            TypeError: If interpolation is not a string
            TypeError: If min_max is not a tuple of two floats
            TypeError: If variables is not a dictionary
            TypeError: If publish is not a list of strings
        """
        # Type check interpolation
        if not isinstance(interpolation, str):
            raise TypeError(f"Interpolation must be a string, got {type(interpolation).__name__}")
            
        # Type check min_max
        if min_max is not None:
            if not isinstance(min_max, tuple) or len(min_max) != 2:
                raise TypeError(f"min_max must be a tuple of two floats, got {type(min_max).__name__}")
            if not all(isinstance(v, (int, float)) for v in min_max):
                raise TypeError(f"min_max values must be numeric (int or float)")
                
        # Type check variables
        if variables is not None and not isinstance(variables, dict):
            raise TypeError(f"Variables must be a dictionary, got {type(variables).__name__}")
            
        # Type check publish
        if publish is not None:
            if not isinstance(publish, list):
                raise TypeError(f"Publish must be a list, got {type(publish).__name__}")
            if not all(isinstance(item, str) for item in publish):
                raise TypeError("All items in publish list must be strings")
                
        self.interpolation = interpolation
        self.min_max = min_max
        self.keyframes: List[Keyframe] = []
        self.variables = variables or {}
        self.publish = publish or []
        self._expression_evaluator = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(interpolation={self.interpolation}, min_max={self.min_max}, keyframes={self.keyframes}, variables={self.variables}, publish={self.publish})"
    
    def add_keyframe(
        self, 
        at: float, 
        value: Union[float, str], 
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ) -> Keyframe:
        """Add a keyframe to this channel.
        
        Args:
            at: The position of this keyframe (0-1 normalized)
            value: The value at this position (number or expression)
            interpolation: Optional interpolation method override for this keyframe
            control_points: Optional control points for bezier interpolation
            derivative: Optional derivative value for hermite interpolation
            
        Returns:
            The created keyframe
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
            # For constant values, create a simple callable that returns a native Python float
            constant_value = float(value)
            value_callable = lambda t, channels={}: constant_value
        elif callable(value):
            # If a callable is already provided, use it directly
            value_callable = value
        else:
            # Try to convert to a string and parse
            try:
                str_value = str(value)
                if self._expression_evaluator is None:
                    from .expression import ExpressionEvaluator
                    self._expression_evaluator = ExpressionEvaluator(self.variables)
                value_callable = self._expression_evaluator.parse_expression(str_value)
            except Exception as e:
                raise TypeError(f"Keyframe value must be a number, string expression, or callable, got {type(value).__name__}: {e}")
        
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
        
        return keyframe
                
    def remove_keyframe(self, at: float) -> None:
        """Remove a keyframe at the specified position.
        
        Args:
            at: The position of the keyframe to remove
            
        Raises:
            ValueError: If no keyframe exists at the specified position
        """
        for i, kf in enumerate(self.keyframes):
            if abs(kf.at - at) < 1e-6:  # Compare with small epsilon for float comparison
                self.keyframes.pop(i)
                return
        
        raise ValueError(f"No keyframe exists at position {at}")
                
    def get_value(self, at: float, channels: Dict[str, float] = None) -> float:
        """Get the interpolated value at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            channels: Optional channel values to use in expressions
            
        Returns:
            The interpolated value at the specified position as a Python float
        """
        if not self.keyframes:
            raise ValueError("Cannot get value: no keyframes defined")
            
        channels = channels or {}
        
        # If position is at or outside the range of keyframes, return the boundary keyframe value
        if at <= self.keyframes[0].at:
            result = self.keyframes[0].value(at, channels)
        elif at >= self.keyframes[-1].at:
            result = self.keyframes[-1].value(at, channels)
        else:
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
            result = self._interpolate(method, at, left_kf, right_kf, channels)
        
        
        # Apply min/max clamping to the final result if specified
        if self.min_max is not None:
            min_val, max_val = self.min_max
            result = max(min_val, min(max_val, result))
        
        # Ensure we always return a Python scalar value
        if hasattr(result, 'item') and hasattr(result, 'size') and result.size == 1:
            return float(result.item())
        else:
            return float(result)
        
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
        if hasattr(left_val, 'item') and hasattr(left_val, 'size') and left_val.size == 1:
            left_val = left_val.item()
        elif hasattr(left_val, 'tolist'):
            left_val = float(left_val)
        else:
            left_val = float(left_val)
            
        if hasattr(right_val, 'item') and hasattr(right_val, 'size') and right_val.size == 1:
            right_val = right_val.item()
        elif hasattr(right_val, 'tolist'):
            right_val = float(right_val)
        else:
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
            List of (position, value) tuples with Python native types
        """
        channels = channels or {}
        
        result = []
        for kf in self.keyframes:
            pos = float(kf.at)
            val = kf.value(kf.at, channels)
            
            # Ensure value is a Python native type
            if hasattr(val, 'item') and hasattr(val, 'size') and val.size == 1:
                val = float(val.item())
            else:
                val = float(val)
                
            result.append((pos, val))
            
        return result
        
    def sample(self, positions: List[float], channels: Dict[str, float] = None) -> List[float]:
        """Sample the channel at multiple positions.
        
        Args:
            positions: List of positions to sample at
            channels: Optional channel values to use in expressions
            
        Returns:
            List of values at the specified positions
        """
        return [self.get_value(at, channels) for at in positions]