"""
Spline class for SplinalTap interpolation.

A Spline represents a single animatable property within a SplineGroup.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .knot import Knot

class Spline:
    """A spline representing a single animatable property within a SplineGroup."""
    
    def __init__(
        self, 
        interpolation: str = "cubic",
        min_max: Optional[Tuple[float, float]] = None,
        variables: Optional[Dict[str, Any]] = None,
        publish: Optional[List[str]] = None
    ):
        """Initialize a spline.
        
        Args:
            interpolation: Default interpolation method for this spline
            min_max: Optional min/max range constraints for this spline's values
            variables: Optional variables to be used in expressions
            publish: Optional list of spline references to publish this spline's value to
            
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
        self.knots: List[Knot] = []
        self.variables = variables or {}
        self.publish = publish or []
        self._expression_evaluator = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(interpolation={self.interpolation}, min_max={self.min_max}, knots={self.knots}, variables={self.variables}, publish={self.publish})"
    
    def add_knot(
        self, 
        at: float, 
        value: Union[float, str], 
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ) -> Knot:
        """Add a knot to this spline.
        
        Args:
            at: The position of this knot (0-1 normalized)
            value: The value at this position (number or expression)
            interpolation: Optional interpolation method override for this knot
            control_points: Optional control points for bezier interpolation
            derivative: Optional derivative value for hermite interpolation
            
        Returns:
            The created knot
        """
        # Validate position range
        if not 0 <= at <= 1:
            raise ValueError(f"Knot position '@' must be between 0 and 1, got {at}")
            
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
                raise TypeError(f"Knot value must be a number, string expression, or callable, got {type(value).__name__}: {e}")
        
        # Create and add the knot
        knot = Knot(at, value_callable, interpolation, control_points, derivative)
        
        # Add to sorted position
        if not self.knots:
            self.knots.append(knot)
        else:
            # Find insertion position
            for i, kf in enumerate(self.knots):
                if at < kf.at:
                    self.knots.insert(i, knot)
                    break
                elif at == kf.at:
                    # Replace existing knot at this position
                    self.knots[i] = knot
                    break
            else:
                # Append at the end if position is greater than all existing knots
                self.knots.append(knot)
        
        return knot
                
    def remove_knot(self, at: float) -> None:
        """Remove a knot at the specified position.
        
        Args:
            at: The position of the knot to remove
            
        Raises:
            ValueError: If no knot exists at the specified position
        """
        for i, kf in enumerate(self.knots):
            if abs(kf.at - at) < 1e-6:  # Compare with small epsilon for float comparison
                self.knots.pop(i)
                return
        
        raise ValueError(f"No knot exists at position {at}")
        
    def get_knot(self, at: float) -> Knot:
        """Get a knot at the specified position.
        
        Args:
            at: The position of the knot to get
            
        Returns:
            The knot at the specified position
            
        Raises:
            ValueError: If no knot exists at the specified position
        """
        for knot in self.knots:
            if abs(knot.at - at) < 1e-6:  # Compare with small epsilon for float comparison
                return knot
                
        raise ValueError(f"No knot exists at position {at}")
            
    def set_knot(
        self, 
        at: float, 
        value: Union[float, str, Callable],
        interpolation: Optional[str] = None
    ) -> Knot:
        """Set a knot value at the specified position.
        
        Args:
            at: The position of the knot to set
            value: The new value
            interpolation: Optional new interpolation method
            
        Returns:
            The updated knot
        """
        # Try to get existing knot first
        try:
            knot = self.get_knot(at)
            
            # Update value
            if isinstance(value, str):
                # Create expression evaluator if needed
                if self._expression_evaluator is None:
                    from .expression import ExpressionEvaluator
                    self._expression_evaluator = ExpressionEvaluator(self.variables)
                    
                # Parse the expression
                value_callable = self._expression_evaluator.parse_expression(value)
            elif isinstance(value, (int, float)):
                # For constant values, create a simple callable
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
                    raise TypeError(f"Knot value must be a number, string expression, or callable, got {type(value).__name__}: {e}")
                    
            # Update the knot
            knot.value = value_callable
            if interpolation is not None:
                knot.interpolation = interpolation
                
            return knot
            
        except ValueError:
            # If the knot doesn't exist, create a new one
            return self.add_knot(at, value, interpolation)
            
    def get_closest_knot(self, at: float) -> Tuple[Knot, float]:
        """Get the closest knot to the specified position.
        
        Args:
            at: The position to find the closest knot for
            
        Returns:
            Tuple of (knot, distance)
            
        Raises:
            ValueError: If no knots exist
        """
        if not self.knots:
            raise ValueError("No knots exist in this spline")
            
        closest_knot = None
        min_distance = float('inf')
        
        for knot in self.knots:
            distance = abs(knot.at - at)
            if distance < min_distance:
                min_distance = distance
                closest_knot = knot
                
        return closest_knot, min_distance
                
    def get_value(self, at: float, channels: Dict[str, float] = None) -> float:
        """Get the interpolated value at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            channels: Optional channel values to use in expressions
            
        Returns:
            The interpolated value at the specified position as a Python float
        """
        if not self.knots:
            raise ValueError("Cannot get value: no knots defined")
            
        channels = channels or {}
        
        # Check if we have a spline_lookup function in the channels
        spline_lookup = None
        if '__spline_lookup__' in channels:
            spline_lookup = channels['__spline_lookup__']
        
        # If position is at or outside the range of knots, return the boundary knot value
        if at <= self.knots[0].at:
            # Special handling for spline lookup in expression values
            if spline_lookup is not None and isinstance(self.knots[0].value, str):
                # Direct evaluation of the expression with spline lookup
                try:
                    return self.knots[0].value(at, channels)
                except Exception as e:
                    # For backward compatibility - try original implementation
                    result = self.knots[0].value(at, channels)
            else:
                result = self.knots[0].value(at, channels)
        elif at >= self.knots[-1].at:
            result = self.knots[-1].value(at, channels)
        else:
            # Find the bracketing knots
            left_kf = None
            right_kf = None
            
            for i in range(len(self.knots) - 1):
                if self.knots[i].at <= at <= self.knots[i + 1].at:
                    left_kf = self.knots[i]
                    right_kf = self.knots[i + 1]
                    break
                    
            if left_kf is None or right_kf is None:
                raise ValueError(f"Could not find bracketing knots for position {at}")
                
            # If the right knot has a specific interpolation method, use that
            # Otherwise use the spline's default method
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
        left_kf: Knot, 
        right_kf: Knot,
        channels: Dict[str, float]
    ) -> float:
        """Interpolate between two knots using the specified method.
        
        Args:
            method: The interpolation method to use
            at: The position to evaluate
            left_kf: The left bracketing knot
            right_kf: The right bracketing knot
            channels: Channel values to use in expressions
            
        Returns:
            The interpolated value
        """
        # Normalize position between the knots
        t_range = right_kf.at - left_kf.at
        if t_range <= 0:
            return left_kf.value(at, channels)
            
        t_norm = (at - left_kf.at) / t_range
        
        # Get knot values and convert to float if needed
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
            # For cubic interpolation, we need more knots ideally
            # This is a simplified implementation
            # Get knots for context
            kfs = self.knots
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
        """Estimate the derivative at a knot position.
        
        Args:
            idx: The index of the knot in the knots list
            
        Returns:
            Estimated derivative value
        """
        kfs = self.knots
        
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
    
    def get_knot_values(self, channels: Dict[str, float] = None) -> List[Tuple[float, float]]:
        """Get all knot positions and values.
        
        Args:
            channels: Optional channel values to use in expressions
            
        Returns:
            List of (position, value) tuples with Python native types
        """
        channels = channels or {}
        
        result = []
        for kf in self.knots:
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
        """Sample the spline at multiple positions.
        
        Args:
            positions: List of positions to sample at
            channels: Optional channel values to use in expressions
            
        Returns:
            List of values at the specified positions
        """
        return [self.get_value(at, channels) for at in positions]
        
    def get_plot(
        self,
        samples: Optional[int] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        show_derivatives: bool = False,
        show_control_points: bool = True
    ):
        """Generate a plot of this spline.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            theme: Plot theme - 'light', 'medium', or 'dark'
            title: Optional title for the plot
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 8)
            height: Optional figure height in inches (defaults to 5)
            show_derivatives: Whether to show derivative vectors for hermite knots
            show_control_points: Whether to show control points for bezier knots
            
        Returns:
            The matplotlib figure
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        # Default number of samples
        if samples is None:
            samples = 100
            
        # Generate sample positions
        positions = [i / (samples - 1) for i in range(samples)]
        
        # Sample the spline
        values = self.sample(positions)
        
        # Set default figure size if not provided
        figure_width = width or 8
        figure_height = height or 5
        
        # Set color based on theme
        if theme == "dark":
            plt.style.use('dark_background')
            line_color = '#ff9500'
            knot_color = '#00b9f1'
            grid_color = '#444444'
            plt.rcParams.update({
                'text.color': '#ffffff',
                'axes.labelcolor': '#ffffff',
                'axes.edgecolor': '#444444',
                'axes.facecolor': '#121212',
                'figure.facecolor': '#121212',
                'grid.color': '#444444',
                'xtick.color': '#aaaaaa',
                'ytick.color': '#aaaaaa',
                'figure.edgecolor': '#121212',
                'savefig.facecolor': '#121212',
                'savefig.edgecolor': '#121212',
                'legend.facecolor': '#121212',
                'legend.edgecolor': '#444444',
                'patch.edgecolor': '#444444'
            })
        elif theme == "medium":
            plt.style.use('default')  # Base on default style
            line_color = '#ff9500'
            knot_color = '#00b9f1'
            grid_color = '#666666'
            plt.rcParams.update({
                'text.color': '#e0e0e0',
                'axes.labelcolor': '#e0e0e0',
                'axes.edgecolor': '#666666',
                'axes.facecolor': '#333333',
                'figure.facecolor': '#222222',
                'grid.color': '#666666',
                'xtick.color': '#cccccc',
                'ytick.color': '#cccccc',
                'figure.edgecolor': '#222222',
                'savefig.facecolor': '#222222',
                'savefig.edgecolor': '#222222',
                'legend.facecolor': '#333333',
                'legend.edgecolor': '#666666',
                'patch.edgecolor': '#666666'
            })
        else:  # light theme
            plt.style.use('default')
            line_color = '#1f77b4'
            knot_color = '#ff7f0e'
            grid_color = 'lightgray'
            plt.rcParams.update({
                'text.color': '#333333',
                'axes.labelcolor': '#333333',
                'axes.edgecolor': '#bbbbbb',
                'axes.facecolor': '#ffffff',
                'figure.facecolor': '#ffffff',
                'grid.color': '#dddddd',
                'xtick.color': '#666666',
                'ytick.color': '#666666',
                'figure.edgecolor': '#ffffff',
                'savefig.facecolor': '#ffffff',
                'savefig.edgecolor': '#ffffff',
                'legend.facecolor': '#ffffff',
                'legend.edgecolor': '#cccccc',
                'patch.edgecolor': '#cccccc'
            })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        
        # Plot spline
        ax.plot(positions, values, label=f"Spline ({self.interpolation})", color=line_color, linewidth=2)
        
        # Get knot positions and values
        knot_positions = []
        knot_values = []
        
        for knot in self.knots:
            knot_positions.append(knot.at)
            
            # Get value at this position
            val = knot.value(knot.at, {})
            
            # Ensure value is a Python native type
            if hasattr(val, 'item') and hasattr(val, 'size') and val.size == 1:
                val = float(val.item())
            else:
                val = float(val)
                
            knot_values.append(val)
        
        # Plot knots
        ax.scatter(knot_positions, knot_values, color=knot_color, s=60, zorder=5, label="Knots")
        
        # Plot additional knot information based on interpolation methods
        if show_derivatives or show_control_points:
            for i, knot in enumerate(self.knots):
                # Show derivatives for hermite knots
                if show_derivatives and knot.derivative is not None:
                    # Draw a tangent line
                    x_range = 0.1  # Draw tangent line extending 0.1 in each direction
                    x_start = max(0, knot.at - x_range)
                    x_end = min(1, knot.at + x_range)
                    
                    val = knot_values[i]
                    y_start = val - knot.derivative * (knot.at - x_start)
                    y_end = val + knot.derivative * (x_end - knot.at)
                    
                    ax.plot([x_start, x_end], [y_start, y_end], color='green', linestyle='--', alpha=0.7, zorder=4)
                
                # Show control points for bezier knots
                if show_control_points and knot.control_points and len(knot.control_points) >= 4:
                    # Only show for right control points in each segment
                    if i > 0:  # Not the first knot
                        cp = knot.control_points
                        
                        # Plot control points
                        ax.scatter([cp[0], cp[2]], [cp[1], cp[3]], color='gray', s=30, alpha=0.7, zorder=3)
                        
                        # Plot control polyline
                        prev_knot = self.knots[i-1]
                        prev_val = knot_values[i-1]
                        ax.plot([prev_knot.at, cp[0], cp[2], knot.at], 
                                [prev_val, cp[1], cp[3], knot_values[i]], 
                                color='gray', linestyle='--', alpha=0.5, zorder=2)
        
        # Set labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Spline with {len(self.knots)} knots")
            
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        ax.legend()
        
        # Set x-axis to 0-1 range
        ax.set_xlim(0, 1)
        
        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def save_plot(
        self,
        filepath: str,
        samples: Optional[int] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        show_derivatives: bool = False,
        show_control_points: bool = True
    ) -> None:
        """Save a plot of this spline to a file.
        
        Args:
            filepath: The file path to save the plot to (e.g., 'plot.png')
            samples: Number of evenly spaced samples to use (defaults to 100)
            theme: Plot theme - 'light', 'medium', or 'dark'
            title: Optional title for the plot
            width: Optional figure width in inches (defaults to 8)
            height: Optional figure height in inches (defaults to 5)
            show_derivatives: Whether to show derivative vectors for hermite knots
            show_control_points: Whether to show control points for bezier knots
            
        Raises:
            ImportError: If matplotlib is not available
        """
        # Get the plot and save it
        self.get_plot(samples, theme, title, save_path=filepath, width=width, height=height,
                     show_derivatives=show_derivatives, show_control_points=show_control_points)
    
    def plot(
        self,
        samples: Optional[int] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        show_derivatives: bool = False,
        show_control_points: bool = True
    ):
        """Plot this spline.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            theme: Plot theme - 'light', 'medium', or 'dark'
            title: Optional title for the plot
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 8)
            height: Optional figure height in inches (defaults to 5)
            show_derivatives: Whether to show derivative vectors for hermite knots
            show_control_points: Whether to show control points for bezier knots
            
        Returns:
            None - displays the plot
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        fig = self.get_plot(samples, theme, title, save_path, width, height, 
                           show_derivatives, show_control_points)
        plt.show()
        return None
    
    def show(
        self,
        samples: Optional[int] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        show_derivatives: bool = False,
        show_control_points: bool = True
    ):
        """Display the plot (alias for plot method).
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            theme: Plot theme - 'light', 'medium', or 'dark'
            title: Optional title for the plot
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 8)
            height: Optional figure height in inches (defaults to 5)
            show_derivatives: Whether to show derivative vectors for hermite knots
            show_control_points: Whether to show control points for bezier knots
        """
        self.plot(samples, theme, title, save_path, width, height, 
                 show_derivatives, show_control_points)
        
    # Backward compatibility methods for Channel
    
    def add_keyframe(
        self, 
        at: float, 
        value: Union[float, str, Callable],
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ) -> Knot:
        """Backward compatibility method that adds a knot to the spline.
        
        Args:
            at: The position of the keyframe (0-1 normalized)
            value: The value at this position (number, expression, or callable)
            interpolation: Optional interpolation method for this keyframe
            control_points: Optional control points for bezier interpolation
            derivative: Optional derivative value for hermite interpolation
            
        Returns:
            The created keyframe (knot)
        """
        return self.add_knot(at, value, interpolation, control_points, derivative)
        
    def remove_keyframe(self, at: float) -> None:
        """Backward compatibility method that removes a knot at the specified position.
        
        Args:
            at: The position of the keyframe to remove
        """
        self.remove_knot(at)
        
    def set_keyframe(
        self, 
        at: float, 
        value: Union[float, str, Callable],
        interpolation: Optional[str] = None
    ) -> Knot:
        """Backward compatibility method that sets a knot value at the specified position.
        
        Args:
            at: The position of the keyframe to set
            value: The new value
            interpolation: Optional new interpolation method
            
        Returns:
            The updated keyframe (knot)
        """
        return self.set_knot(at, value, interpolation)
        
    def get_keyframe(self, at: float) -> Knot:
        """Backward compatibility method that gets a knot at the specified position.
        
        Args:
            at: The position of the keyframe to get
            
        Returns:
            The keyframe (knot) at the specified position
        """
        return self.get_knot(at)
        
    @property
    def keyframes(self) -> List[Knot]:
        """Backward compatibility property that returns the knots list.
        
        Returns:
            List of keyframes (knots)
        """
        return self.knots
        
    def get_closest_keyframe(self, at: float) -> Tuple[Knot, float]:
        """Backward compatibility method that gets the closest knot.
        
        Args:
            at: The position to find the closest keyframe for
            
        Returns:
            Tuple of (keyframe, distance)
        """
        return self.get_closest_knot(at)
        
    def get_keyframe_values(self, channels: Dict[str, float] = None) -> List[Tuple[float, float]]:
        """Backward compatibility method that gets all keyframe positions and values.
        
        Args:
            channels: Optional channel values to use in expressions
            
        Returns:
            List of (position, value) tuples with Python native types
        """
        return self.get_knot_values(channels)
        
    def get_knot_values(self, channels: Dict[str, float] = None) -> List[Tuple[float, float]]:
        """Get all knot positions and values.
        
        Args:
            channels: Optional channel values to use in expressions
            
        Returns:
            List of (position, value) tuples with Python native types
        """
        channels = channels or {}
        
        result = []
        for kf in self.knots:
            pos = float(kf.at)
            val = kf.value(kf.at, channels)
            
            # Ensure value is a Python native type
            if hasattr(val, 'item') and hasattr(val, 'size') and val.size == 1:
                val = float(val.item())
            else:
                val = float(val)
                
            result.append((pos, val))
            
        return result