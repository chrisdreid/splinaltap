"""Knot class for SplinalTap interpolation.

A Knot represents a single point on a Spline with a position, value, 
and interpolation method."""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math

class Knot:
    """A knot with position, value, interpolation method, and additional parameters."""
    
    def __init__(
        self, 
        at: float, 
        value: Union[float, str, Callable],
        interpolation: Optional[str] = None,
        control_points: Optional[List[float]] = None,
        derivative: Optional[float] = None
    ):
        """Initialize a knot.
        
        Args:
            at: The position of this knot (0-1 normalized)
            value: The value at this position (number, expression, or callable)
            interpolation: Optional interpolation method for this knot
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
        return f"Knot(at={self.at}, value={self.value}, interpolation={self.interpolation})"

    def get_plot(self, samples: int = 100, theme: str = "dark", width: Optional[float] = None, height: Optional[float] = None, title: Optional[str] = None, save_path: Optional[str] = None):
        """Generate a plot of this knot.
        
        Args:
            samples: Number of samples to use for the plot
            theme: Plot theme (dark, medium, or light)
            width: Optional figure width in inches
            height: Optional figure height in inches
            title: Optional plot title
            save_path: Optional path to save the plot
        
        Returns:
            The matplotlib figure
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")

        # Set up theme
        if theme == "dark":
            plt.style.use('dark_background')
            grid_color = '#444444'
            point_color = '#ff9500'
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
            grid_color = '#666666'
            point_color = '#ff9500'
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
            grid_color = 'lightgray'
            point_color = '#1f77b4'
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
            
        # Set default figure size if not provided
        figure_width = width or 6
        figure_height = height or 4
        
        # Create figure
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))

        # Plot the knot point
        val = self.value
        if callable(val):
            try:
                val = val(self.at, {})
            except:
                val = 0
                
        # Ensure value is a Python scalar
        try:
            if hasattr(val, 'item'):
                val = float(val.item())
            else:
                val = float(val)
        except:
            val = 0
            
        # Plot knot point
        ax.scatter([self.at], [val], color=point_color, s=80, zorder=5)
        
        # Show the control points for bezier
        if self.control_points and len(self.control_points) >= 4:
            # Extract control points [p1_x, p1_y, p2_x, p2_y]
            cp = self.control_points
            ax.scatter([cp[0], cp[2]], [cp[1], cp[3]], color='gray', s=40, alpha=0.7, zorder=3)
            ax.plot([self.at, cp[0], cp[2], 1.0], [val, cp[1], cp[3], 0], color='gray', linestyle='--', alpha=0.5, zorder=2)

        # Show derivatives for hermite
        if self.derivative is not None:
            # Draw a tangent line
            x_range = 0.2  # Draw tangent line extending 0.2 in each direction
            x_start = max(0, self.at - x_range)
            x_end = min(1, self.at + x_range)
            y_start = val - self.derivative * (self.at - x_start)
            y_end = val + self.derivative * (x_end - self.at)
            ax.plot([x_start, x_end], [y_start, y_end], color='green', linestyle='--', alpha=0.7, zorder=4)
            
        # Add labels
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Knot at {self.at} with value {val:.2f}")
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        
        # Set axis ranges
        ax.set_xlim(0, 1)
        
        # Add annotation
        ax.annotate(f"({self.at:.2f}, {val:.2f})", 
                    xy=(self.at, val), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
        
        if self.interpolation:
            ax.text(0.02, 0.02, f"Interpolation: {self.interpolation}", 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='bottom')
            
        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot(self, samples: int = 100, theme: str = "dark", width: Optional[float] = None, height: Optional[float] = None, title: Optional[str] = None, save_path: Optional[str] = None):
        """Plot this knot.
        
        Args:
            samples: Number of samples to use for the plot
            theme: Plot theme (dark, medium, or light)
            width: Optional figure width in inches
            height: Optional figure height in inches
            title: Optional plot title
            save_path: Optional path to save the plot
            
        Returns:
            None - displays the plot
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        fig = self.get_plot(samples, theme, width, height, title, save_path)
        plt.show()
        return None
        
    def show(self, samples: int = 100, theme: str = "dark", width: Optional[float] = None, height: Optional[float] = None, title: Optional[str] = None, save_path: Optional[str] = None):
        """Display the plot (alias for plot method).
        
        Args:
            samples: Number of samples to use for the plot
            theme: Plot theme (dark, medium, or light)
            width: Optional figure width in inches
            height: Optional figure height in inches
            title: Optional plot title
            save_path: Optional path to save the plot
        """
        self.plot(samples, theme, width, height, title, save_path)
        
    def save_plot(self, filepath: str, samples: int = 100, theme: str = "dark", width: Optional[float] = None, height: Optional[float] = None, title: Optional[str] = None):
        """Save a plot of this knot to a file.
        
        Args:
            filepath: Path to save the plot
            samples: Number of samples to use for the plot
            theme: Plot theme (dark, medium, or light)
            width: Optional figure width in inches
            height: Optional figure height in inches
            title: Optional plot title
            
        Raises:
            ImportError: If matplotlib is not available
        """
        self.get_plot(samples, theme, width, height, title, save_path=filepath)
