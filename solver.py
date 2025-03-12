"""
KeyframeSolver class for SplinalTap interpolation.

A KeyframeSolver is a collection of Splines that can be evaluated together.
It represents a complete animation or property set, like a scene in 3D software.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple

from .spline import Spline
from .expression import ExpressionEvaluator

# KeyframeSolver file format version for compatibility checking
KEYFRAME_SOLVER_FORMAT_VERSION = "1.0"

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


class KeyframeSolver:
    """A solver containing multiple splines for complex animation."""
    
    def __init__(self, name: str = "Untitled"):
        """Initialize a new solver.
        
        Args:
            name: The name of the solver
        """
        self.name = name
        self.splines: Dict[str, Spline] = {}
        self.metadata: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.range: Tuple[float, float] = (0.0, 1.0)
    
    def create_spline(self, name: str) -> Spline:
        """Create a new spline in this solver.
        
        Args:
            name: The name of the spline
            
        Returns:
            The newly created spline
        """
        spline = Spline()
        self.splines[name] = spline
        return spline
    
    def get_spline(self, name: str) -> Spline:
        """Get a spline by name.
        
        Args:
            name: The name of the spline to get
            
        Returns:
            The requested spline
            
        Raises:
            KeyError: If the spline does not exist
        """
        if name not in self.splines:
            raise KeyError(f"Spline '{name}' does not exist in this solver")
        return self.splines[name]
    
    def get_spline_names(self) -> List[str]:
        """Get the names of all splines in this solver.
        
        Returns:
            A list of spline names
        """
        return list(self.splines.keys())
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value.
        
        Args:
            name: The variable name
            value: The variable value
        """
        self.variables[name] = value
    
    def solve(self, position: float, external_channels: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Solve all splines at a specific position.
        
        Args:
            position: The position to solve at
            external_channels: Optional external channel values
            
        Returns:
            A dictionary of spline names to channel value dictionaries
        """
        result = {}
        
        for spline_name, spline in self.splines.items():
            # Create a result dictionary for this spline
            spline_result = {}
            
            # Evaluate each channel
            for channel_name, channel in spline.channels.items():
                # Combine variables with external channels
                combined_channels = {}
                if external_channels:
                    combined_channels.update(external_channels)
                combined_channels.update(self.variables)
                
                # Evaluate the channel at the given position
                value = channel.get_value(position, combined_channels)
                spline_result[channel_name] = value
            
            # Add the result to the main result dictionary
            result[spline_name] = spline_result
        
        return result
    
    def save(self, filepath: str, format: Optional[str] = None) -> None:
        """Save the solver to a file.
        
        Args:
            filepath: The path to save to
            format: The format to save in (json, pickle, yaml, or numpy)
        """
        # Determine format from extension if not specified
        if format is None:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext == '.pkl':
                format = 'pickle'
            elif ext == '.yaml' or ext == '.yml':
                format = 'yaml'
            elif ext == '.npz':
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
        
        # Convert to dictionary representation
        data = self._serialize()
        
        # Save in the appropriate format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML support")
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy support")
            metadata = json.dumps(data)
            np.savez(filepath, metadata=metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the solver to a dictionary.
        
        Returns:
            Dictionary representation of the solver
        """
        # Start with basic information
        data = {
            "version": KEYFRAME_SOLVER_FORMAT_VERSION,
            "name": self.name,
            "metadata": self.metadata,
            "range": self.range,
            "variables": {}
        }
        
        # Add variables (with conversion for NumPy types)
        for name, value in self.variables.items():
            if HAS_NUMPY and isinstance(value, np.ndarray):
                data["variables"][name] = value.tolist()
            elif HAS_NUMPY and isinstance(value, np.number):
                data["variables"][name] = float(value)
            else:
                data["variables"][name] = value
        
        # Add splines
        data["splines"] = {}
        for spline_name, spline in self.splines.items():
            # Create a dictionary for this spline
            spline_data = {
                "channels": {}
            }
            
            # Add channels
            for channel_name, channel in spline.channels.items():
                # Create a dictionary for this channel
                channel_data = {
                    "interpolation": channel.interpolation,
                    "keyframes": []
                }
                
                # Add min/max if set
                if channel.min_max is not None:
                    channel_data["min_max"] = channel.min_max
                
                # Add keyframes
                for keyframe in channel.keyframes:
                    # Create a dictionary for this keyframe
                    # Convert function values to strings to avoid serialization errors
                    # We need to handle the fact that keyframe.value is a callable
                    # Let's try to convert it to a string representation if possible
                    value = None
                    
                    if isinstance(keyframe.value, (int, float)):
                        value = keyframe.value
                    elif isinstance(keyframe.value, str):
                        value = keyframe.value
                    else:
                        # This is probably a callable, so we'll just use a string representation
                        value = "0"  # Default fallback
                    
                    keyframe_data = {
                        "position": keyframe.at,
                        "value": value
                    }
                    
                    # Add interpolation if different from channel default
                    if keyframe.interpolation is not None:
                        keyframe_data["interpolation"] = keyframe.interpolation
                    
                    # Add parameters
                    params = {}
                    if keyframe.derivative is not None:
                        params["deriv"] = keyframe.derivative
                    if keyframe.control_points is not None:
                        params["cp"] = keyframe.control_points
                    if params:
                        keyframe_data["parameters"] = params
                    
                    # Add this keyframe to the channel data
                    channel_data["keyframes"].append(keyframe_data)
                
                # Add this channel to the spline data
                spline_data["channels"][channel_name] = channel_data
            
            # Add this spline to the data
            data["splines"][spline_name] = spline_data
        
        return data
    
    @classmethod
    def load(cls, filepath: str, format: Optional[str] = None) -> 'KeyframeSolver':
        """Load a solver from a file.
        
        Args:
            filepath: The path to load from
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
            
        Returns:
            The loaded Solver
        """
        # Determine format from extension if not specified
        if format is None:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext == '.pkl':
                format = 'pickle'
            elif ext == '.yaml' or ext == '.yml':
                format = 'yaml'
            elif ext == '.npz':
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
        
        # Load based on format
        if format == 'json':
            with open(filepath, 'r') as f:
                solver_data = json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                solver_data = pickle.load(f)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML support")
            with open(filepath, 'r') as f:
                solver_data = yaml.safe_load(f)
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy support")
            np_data = np.load(filepath)
            solver_data = json.loads(np_data['metadata'].item())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return cls._deserialize(solver_data)
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> 'KeyframeSolver':
        """Deserialize a solver from a dictionary.
        
        Args:
            data: Dictionary representation of the solver
            
        Returns:
            The deserialized Solver
        """
        # Check version if available
        if "version" in data:
            file_version = data["version"]
            if file_version != KEYFRAME_SOLVER_FORMAT_VERSION:
                print(f"Warning: KeyframeSolver file version ({file_version}) does not match current version ({KEYFRAME_SOLVER_FORMAT_VERSION}). Some features may not work correctly.")
        
        # Create a new solver
        solver = cls(name=data.get("name", "Untitled"))
        
        # Set range
        if "range" in data:
            solver.range = tuple(data["range"])
        
        # Set metadata
        solver.metadata = data.get("metadata", {})
        
        # Set variables
        for name, value in data.get("variables", {}).items():
            solver.set_variable(name, value)
        
        # Create splines
        for spline_name, spline_data in data.get("splines", {}).items():
            # Create a new spline
            spline = solver.create_spline(spline_name)
            
            # Check if there's a 'channels' key in the spline data (new format)
            channels_data = spline_data.get("channels", {})
            if channels_data:
                # Process each channel in the channels dictionary
                for channel_name, channel_data in channels_data.items():
                    # Create a new channel
                    interpolation = channel_data.get("interpolation", "cubic")
                    min_max = channel_data.get("min_max")
                    
                    channel = spline.add_channel(
                        name=channel_name,
                        interpolation=interpolation,
                        min_max=min_max
                    )
                    
                    # Add keyframes to this channel
                    for keyframe_data in channel_data.get("keyframes", []):
                        position = keyframe_data.get("position", 0)
                        value = keyframe_data.get("value", 0)
                        interp = keyframe_data.get("interpolation")
                        params = keyframe_data.get("parameters", {})
                        
                        control_points = None
                        derivative = None
                        
                        if params:
                            if "cp" in params:
                                control_points = params["cp"]
                            if "deriv" in params:
                                derivative = params["deriv"]
                        
                        channel.add_keyframe(
                            at=position,
                            value=value,
                            interpolation=interp,
                            control_points=control_points,
                            derivative=derivative
                        )
            else:
                # Legacy format - channel data directly in spline
                for channel_name, channel_data in spline_data.items():
                    # Create a new channel
                    interpolation = channel_data.get("interpolation", "cubic")
                    min_max = channel_data.get("min_max")
                    
                    channel = spline.add_channel(
                        name=channel_name,
                        interpolation=interpolation,
                        min_max=min_max
                    )
                    
                    # Add keyframes
                    for keyframe_data in channel_data.get("keyframes", []):
                        position = keyframe_data.get("position", 0)
                        value = keyframe_data.get("value", 0)
                        interp = keyframe_data.get("interpolation")
                        params = keyframe_data.get("parameters", {})
                        
                        control_points = None
                        derivative = None
                        
                        if params:
                            if "cp" in params:
                                control_points = params["cp"]
                            if "deriv" in params:
                                derivative = params["deriv"]
                        
                        channel.add_keyframe(
                            at=position,
                            value=value,
                            interpolation=interp,
                            control_points=control_points,
                            derivative=derivative
                        )
        
        return solver