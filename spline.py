"""
Spline class for SplinalTap interpolation.

A Spline represents a complete curve or property composed of multiple channels.
For example, a "position" spline might have "x", "y", and "z" channels.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

from .channel import Channel
from .expression import ExpressionEvaluator


class Spline:
    """A spline representing a complete curve with multiple channels."""
    
    def __init__(
        self, 
        range: Optional[Tuple[float, float]] = None,
        variables: Optional[Dict[str, Any]] = None
    ):
        """Initialize a spline.
        
        Args:
            range: Optional global time range [min, max] for the spline
            variables: Optional variables to be used in expressions
        """
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
        if name in self.channels and not replace:
            return self.channels[name]
            raise ValueError(f"Channel '{name}' already exists in this spline")
        
            
        # Create a new channel with the shared variables
        channel = Channel(
            interpolation=interpolation,
            min_max=min_max,
            variables=self.variables,
            publish=publish
        )
        
        self.channels[name] = channel
        return channel
    
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