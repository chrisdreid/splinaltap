"""
Spline class for SplinalTap interpolation.

A Spline represents a complete curve or property composed of multiple channels.
For example, a "position" spline might have "x", "y", and "z" channels.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .channel import Channel
from .expression import ExpressionEvaluator


class Spline:
    """A spline representing a complete curve with multiple channels."""
    
    def __init__(
        self, 
        range: Optional[Tuple[float, float]] = None,
        variables: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """Initialize a spline.
        
        Args:
            range: Optional global time range [min, max] for the spline
            variables: Optional variables to be used in expressions
            callbacks: Optional callbacks to be called on channel access
            
        Raises:
            TypeError: If range is not a tuple of two floats
            TypeError: If variables is not a dictionary
            TypeError: If callbacks is not a dictionary
        """
        # Type check range
        if range is not None:
            if not isinstance(range, tuple) or len(range) != 2:
                raise TypeError(f"Range must be a tuple of two floats, got {type(range).__name__}")
            if not all(isinstance(v, (int, float)) for v in range):
                raise TypeError(f"Range values must be numeric (int or float)")
                
        # Type check variables
        if variables is not None and not isinstance(variables, dict):
            raise TypeError(f"Variables must be a dictionary, got {type(variables).__name__}")
            
        # Type check callbacks
        if callbacks is not None and not isinstance(callbacks, dict):
            raise TypeError(f"Callbacks must be a dictionary, got {type(callbacks).__name__}")
        
        self.range = range or (0.0, 1.0)
        self.variables = variables or {}
        self.channels: Dict[str, Channel] = {}
        self._expression_evaluator = ExpressionEvaluator(self.variables)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(range={self.range}, variables={self.variables}, channels={self.channels})"
    
    def add_channel(
        self, 
        name: str, 
        interpolation: str = "cubic",
        min_max: Optional[Tuple[float, float]] = None,
        replace: bool = False,
        publish: Optional[List[str]] = None
    ) -> Channel:
        """Add a new channel to this spline.
        
        Args:
            name: The channel name
            interpolation: Default interpolation method for this channel
            min_max: Optional min/max range constraints for this channel's values
            replace: If True, replace existing channel with the same name
            publish: Optional list of channel references to publish this channel's value to
            
        Returns:
            The newly created channel
        """
        if name in self.channels:
            if not replace:
                raise ValueError(f"Channel '{name}' already exists in this spline")
            return self.channels[name]
        
            
        # Create a new channel with the shared variables
        channel = Channel(
            interpolation=interpolation,
            min_max=min_max,
            variables=self.variables,
            publish=publish
        )
        
        self.channels[name] = channel
        return channel
    
    def set_publish(self, channel_name: str, publish: List[str]) -> None:
        """Set the publish directive for a channel.
        
        Args:
            channel_name: The channel name
            publish: The publish directive
        """
        self.channels[channel_name].publish = publish
        # needs to send callback up to the solver
        
        
    def get_channel(self, name: str) -> Channel:
        """Get a channel by name.
        
        Args:
            name: The channel name
            
        Returns:
            The channel object
        """
        if name not in self.channels:
            raise ValueError(f"Channel '{name}' does not exist in this spline")
            
        return self.channels[name]
    
    def set_keyframe(
        self, 
        at: float, 
        values: Dict[str, Union[float, str]],
        interpolation: Optional[str] = None
    ) -> None:
        """Set keyframes across multiple channels simultaneously.
        
        Args:
            at: The position to set keyframes at (0-1 normalized)
            values: Dictionary of channel name to value
            interpolation: Optional interpolation method for all channels
        """
        for channel_name, value in values.items():
            # Create channel if it doesn't exist
            if channel_name not in self.channels:
                self.add_channel(channel_name)
                
            # Add keyframe to the channel
            self.channels[channel_name].add_keyframe(at, value, interpolation)
    
    def get_value(
        self, 
        at: float, 
        channel_names: Optional[List[str]] = None,
        ext_channels: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Get values from multiple channels at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            channel_names: Optional list of channel names to get (all if None)
            ext_channels: Optional external channel values to use in expressions
            
        Returns:
            Dictionary of channel name to interpolated value
        """
        result = {}
        
        # Apply range normalization
        min_t, max_t = self.range
        at_scaled = min_t + at * (max_t - min_t)
        
        # Get the channels to evaluate
        channels_to_eval = channel_names or list(self.channels.keys())
        
        # Evaluate each channel
        for name in channels_to_eval:
            if name in self.channels:
                value = self.channels[name].get_value(at, ext_channels)
                
                # Convert numpy arrays to Python float
                if hasattr(value, 'item') or hasattr(value, 'tolist'):
                    try:
                        if hasattr(value, 'item'):
                            value = float(value.item())
                        else:
                            value = float(value)
                    except:
                        value = float(value)
                
                result[name] = value
            else:
                raise ValueError(f"Channel '{name}' does not exist in this spline")
                
        return result
    
    def get_channel_value(
        self, 
        channel_name: str, 
        at: float,
        ext_channels: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get a single channel value at the specified position.
        
        Args:
            channel_name: The channel name
            at: The position to evaluate (0-1 normalized)
            ext_channels: Optional external channel values to use in expressions
            
        Returns:
            The interpolated value for the specified channel
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' does not exist in this spline")
            
        # Apply range normalization
        min_t, max_t = self.range
        at_scaled = min_t + at * (max_t - min_t)
        
        return self.channels[channel_name].get_value(at_scaled, ext_channels)
    
    def set_variable(self, name: str, value: Union[float, str]) -> None:
        """Set a variable for use in expressions.
        
        Args:
            name: The variable name
            value: The variable value (number or expression)
        """
        if isinstance(value, str):
            # Parse the expression
            self.variables[name] = self._expression_evaluator.parse_expression(value)
        else:
            # Store the value directly
            self.variables[name] = value
            
        # Update all channels with the new variable
        for channel in self.channels.values():
            channel.variables = self.variables
    
    def get_keyframe_positions(self) -> List[float]:
        """Get a sorted list of all unique keyframe positions across all channels.
        
        Returns:
            List of unique keyframe positions
        """
        positions = set()
        
        for channel in self.channels.values():
            for keyframe in channel.keyframes:
                positions.add(keyframe.at)
                
        return sorted(positions)
    
    def sample(
        self, 
        positions: List[float],
        channel_names: Optional[List[str]] = None,
        ext_channels: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """Sample multiple channels at specified positions.
        
        Args:
            positions: List of positions to sample at
            channel_names: Optional list of channel names to sample (all if None)
            ext_channels: Optional external channel values to use in expressions
            
        Returns:
            Dictionary of channel name to list of sampled values
        """
        # Get the channels to sample
        channels_to_sample = channel_names or list(self.channels.keys())
        
        # Initialize results
        results: Dict[str, List[float]] = {name: [] for name in channels_to_sample}
        
        # Sample each position
        for at in positions:
            channel_values = self.get_value(at, channels_to_sample, ext_channels)
            
            for name, value in channel_values.items():
                results[name].append(value)
                
        return results
    
    def linspace(
        self, 
        num_samples: int,
        channel_names: Optional[List[str]] = None,
        ext_channels: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """Sample channels at evenly spaced positions.
        
        Args:
            num_samples: Number of samples to generate
            channel_names: Optional list of channel names to sample (all if None)
            ext_channels: Optional external channel values to use in expressions
            
        Returns:
            Dictionary of channel name to list of sampled values
        """
        if num_samples < 2:
            raise ValueError("Number of samples must be at least 2")
            
        # Generate evenly spaced positions
        positions = [i / (num_samples - 1) for i in range(num_samples)]
        
        # Sample at these positions
        return self.sample(positions, channel_names, ext_channels)
        
    def get_plot(
        self,
        samples: Optional[int] = None,
        filter_channels: Optional[List[str]] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None
    ):
        """Generate a plot of the spline's channels.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional list of channel names to include (all if None)
            theme: Plot theme - 'light' or 'dark'
            title: Optional title for the plot (defaults to spline name if available)
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 10)
            height: Optional figure height in inches (defaults to 6)
            
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
        
        # Get channel values
        channel_values = self.sample(positions, filter_channels)
        
        # Set default figure size if not provided
        figure_width = width or 10
        figure_height = height or 6
        
        # Create figure
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        
        # Set color based on theme
        if theme == "dark":
            plt.style.use('dark_background')
            color_palette = ['#ff9500', '#00b9f1', '#fb02fe', '#01ff66', '#fffd01', '#ff2301']
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
            color_palette = ['#ff9500', '#00b9f1', '#fb02fe', '#01ff66', '#fffd01', '#ff2301']
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
            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
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
            
        # Plot each channel
        for i, (channel_name, values) in enumerate(channel_values.items()):
            color = color_palette[i % len(color_palette)]
            ax.plot(positions, values, label=channel_name, color=color)
            
            # Add markers at keyframe positions
            keyframe_positions = [kf.at for kf in self.channels[channel_name].keyframes]
            keyframe_values = [self.channels[channel_name].get_value(pos) for pos in keyframe_positions]
            ax.scatter(keyframe_positions, keyframe_values, color=color, s=50)
            
        # Set labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        
        if title:
            ax.set_title(title)
        elif hasattr(self, 'name'):
            ax.set_title(getattr(self, 'name'))
            
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
        
        # Use custom legend style for each theme
        if theme == "dark":
            ax.legend(facecolor='#121212', edgecolor='#444444', labelcolor='white')
        elif theme == "medium":
            ax.legend(facecolor='#333333', edgecolor='#666666', labelcolor='#e0e0e0')
        else:  # light theme
            ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
        
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
        filter_channels: Optional[List[str]] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None
    ) -> None:
        """Save a plot of the spline's channels to a file.
        
        Args:
            filepath: The file path to save the plot to (e.g., 'plot.png')
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional list of channel names to include (all if None)
            theme: Plot theme - 'light' or 'dark'
            title: Optional title for the plot (defaults to spline name if available)
            width: Optional figure width in inches (defaults to 10)
            height: Optional figure height in inches (defaults to 6)
            
        Raises:
            ImportError: If matplotlib is not available
        """
        # Get the plot and save it
        self.get_plot(samples, filter_channels, theme, title, save_path=filepath, width=width, height=height)
        
    def plot(
        self,
        samples: Optional[int] = None,
        filter_channels: Optional[List[str]] = None,
        theme: str = "dark",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None
    ):
        """Plot the spline's channels.
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional list of channel names to include (all if None)
            theme: Plot theme - 'light' or 'dark'
            title: Optional title for the plot (defaults to spline name if available)
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 10)
            height: Optional figure height in inches (defaults to 6)
            
        Returns:
            None - displays the plot
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib")
            
        fig = self.get_plot(samples, filter_channels, theme, title, save_path, width, height)
        plt.show()
        return None
        
    def show(
        self, 
        samples: Optional[int] = None, 
        filter_channels: Optional[List[str]] = None, 
        theme: str = "dark", 
        title: Optional[str] = None, 
        save_path: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None
    ):
        """Display the plot (alias for plot method).
        
        Args:
            samples: Number of evenly spaced samples to use (defaults to 100)
            filter_channels: Optional list of channel names to include (all if None)
            theme: Plot theme - 'light' or 'dark'
            title: Optional title for the plot (defaults to spline name if available)
            save_path: Optional file path to save the plot (e.g., 'plot.png')
            width: Optional figure width in inches (defaults to 10)
            height: Optional figure height in inches (defaults to 6)
        """
        self.plot(samples, filter_channels, theme, title, save_path, width, height)