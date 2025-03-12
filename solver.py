"""
Solver class for SplinalTap interpolation.

A Solver is a collection of Splines that can be evaluated together.
It represents a complete animation or property set, like a scene in 3D software.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple

from .spline import Spline
from .expression import ExpressionEvaluator

# Solver file format version for compatibility checking
SOLVER_FORMAT_VERSION = "1.0"

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


class Solver:
    """A solver containing multiple splines for complex animation."""
    
    def __init__(self, name: str = "Untitled"):
        """Initialize a new solver.
        
        Args:
            name: The name of the solver
        """
        self.name = name
        self.splines: Dict[str, Spline] = {}
        self.metadata: Dict[str, Any] = {}
        self.range: Tuple[float, float] = (0.0, 1.0)
        self.variables: Dict[str, Any] = {}
        self._expression_evaluator = ExpressionEvaluator(self.variables)
    
    def add_spline(self, name: str, spline: Spline) -> None:
        """Add a spline to the solver.
        
        Args:
            name: The name to identify this spline
            spline: The Spline instance to add
        """
        self.splines[name] = spline
    
    def create_spline(
        self, 
        name: str, 
        range: Optional[Tuple[float, float]] = None
    ) -> Spline:
        """Create a new spline and add it to the solver.
        
        Args:
            name: The name for the new spline
            range: Optional time range for the spline (uses solver range if None)
            
        Returns:
            The newly created spline
        """
        if name in self.splines:
            raise ValueError(f"Spline '{name}' already exists in this solver")
            
        # Create a new spline with shared variables
        spline = Spline(
            range=range or self.range,
            variables=self.variables
        )
        
        self.splines[name] = spline
        return spline
    
    def remove_spline(self, name: str) -> None:
        """Remove a spline from the solver.
        
        Args:
            name: The name of the spline to remove
        """
        if name in self.splines:
            del self.splines[name]
        else:
            raise KeyError(f"No spline named '{name}' in solver")
    
    def get_spline(self, name: str) -> Spline:
        """Get a spline by name.
        
        Args:
            name: The name of the spline to get
            
        Returns:
            The Spline instance
        """
        if name in self.splines:
            return self.splines[name]
        else:
            raise KeyError(f"No spline named '{name}' in solver")
    
    def get_spline_names(self) -> List[str]:
        """Get the names of all splines in the solver.
        
        Returns:
            List of spline names
        """
        return list(self.splines.keys())
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value.
        
        Args:
            key: The metadata key
            default: The default value to return if key not found
            
        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)
    
    def set_variable(self, name: str, value: Union[float, str]) -> None:
        """Set a variable for use in expressions across all splines.
        
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
            
        # Update all splines with the new variable
        for spline in self.splines.values():
            spline.variables = self.variables
    
    def solve(
        self, 
        at: float, 
        spline_names: Optional[List[str]] = None,
        channel_names: Optional[Dict[str, List[str]]] = None,
        ext_channels: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Solve all splines at the specified position.
        
        Args:
            at: The position to evaluate (0-1 normalized)
            spline_names: Optional list of spline names to solve (all if None)
            channel_names: Optional dict of spline name to list of channel names
            ext_channels: Optional external channel values to use in expressions
            
        Returns:
            Dictionary of spline name to dict of channel name to value
        """
        result = {}
        
        # Determine which splines to evaluate
        splines_to_eval = spline_names or list(self.splines.keys())
        
        # Evaluate each spline
        for name in splines_to_eval:
            if name in self.splines:
                spline = self.splines[name]
                
                # Get the channels to evaluate for this spline
                spline_channels = None
                if channel_names and name in channel_names:
                    spline_channels = channel_names[name]
                    
                # Evaluate the spline
                result[name] = spline.get_value(at, spline_channels, ext_channels)
            else:
                raise ValueError(f"Spline '{name}' does not exist in this solver")
                
        return result
    
    def save(self, filepath: str, format: Optional[str] = None) -> None:
        """Save the solver to a file.
        
        Args:
            filepath: The path to save to
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
                   If not specified, inferred from the file extension
        """
        # Determine format from file extension if not specified
        if format is None:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            if ext == '.json':
                format = 'json'
            elif ext in ('.pkl', '.pickle'):
                format = 'pickle'
            elif ext in ('.yml', '.yaml'):
                format = 'yaml'
            elif ext == '.npz' and HAS_NUMPY:
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
                
        # Prepare the solver data
        solver_data = self._serialize()
            
        # Save in the requested format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(solver_data, f, indent=2)
                
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(solver_data, f)
                
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
                
            with open(filepath, 'w') as f:
                yaml.dump(solver_data, f, sort_keys=False)
                
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy format. Install with: pip install numpy")
                
            # Convert solver data to a format suitable for NumPy
            np_data = {
                'metadata': json.dumps(solver_data)
            }
            
            # Sample values for fast loading (optional)
            for spline_name, spline in self.splines.items():
                try:
                    # Sample 1000 points for each spline
                    samples = spline.linspace(1000)
                    for channel_name, values in samples.items():
                        np_data[f"{spline_name}.{channel_name}"] = np.array(values)
                except Exception as e:
                    print(f"Warning: Could not sample values for {spline_name}: {e}")
            
            np.savez(filepath, **np_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the solver to a dictionary.
        
        Returns:
            Dictionary representation of the solver
        """
        # Start with basic information
        data = {
            "version": SOLVER_FORMAT_VERSION,
            "name": self.name,
            "metadata": self.metadata,
            "range": self.range,
            "variables": {}
        }
        
        # Serialize variables (convert callable variables to their string representation)
        for name, value in self.variables.items():
            if callable(value):
                # For now, we can't serialize callables - would need to define a format
                data["variables"][name] = str(value)
            else:
                data["variables"][name] = value
        
        # Serialize splines
        data["splines"] = {}
        
        for spline_name, spline in self.splines.items():
            spline_data = {}
            
            # Serialize each channel in the spline
            for channel_name, channel in spline.channels.items():
                channel_data = {
                    "interpolation": channel.interpolation
                }
                
                if channel.min_max:
                    channel_data["min-max"] = channel.min_max
                
                # Serialize keyframes
                keyframes = []
                for keyframe in channel.keyframes:
                    kf_data = {
                        "@": keyframe.at
                    }
                    
                    # Get the value (this is complex as it might be a callable)
                    if callable(keyframe.value):
                        try:
                            # Evaluate at the keyframe position to get the value
                            value = keyframe.value(keyframe.at, {})
                            
                            # Convert numpy arrays to Python float
                            if hasattr(value, 'tolist') or hasattr(value, 'item'):
                                try:
                                    if hasattr(value, 'item'):
                                        value = float(value.item())
                                    else:
                                        value = float(value)
                                except:
                                    value = float(value)
                            
                            # If the result is a simple value, use that
                            kf_data["value"] = value
                        except:
                            # If evaluation fails, it might be an expression
                            kf_data["value"] = "expr()"  # placeholder
                    else:
                        kf_data["value"] = keyframe.value
                    
                    # Add optional parameters
                    if keyframe.interpolation:
                        kf_data["interpolation"] = keyframe.interpolation
                        
                    if keyframe.control_points:
                        kf_data["control-points"] = keyframe.control_points
                        
                    if keyframe.derivative is not None:
                        kf_data["derivative"] = keyframe.derivative
                    
                    keyframes.append(kf_data)
                
                channel_data["keyframes"] = keyframes
                spline_data[channel_name] = channel_data
            
            data["splines"][spline_name] = spline_data
        
        return data
    
    @classmethod
    def load(cls, filepath: str, format: Optional[str] = None) -> 'Solver':
        """Load a solver from a file.
        
        Args:
            filepath: The path to load from
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
                  If not specified, inferred from the file extension
                  
        Returns:
            The loaded Solver
        """
        # Determine format from file extension if not specified
        if format is None:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            if ext == '.json':
                format = 'json'
            elif ext in ('.pkl', '.pickle'):
                format = 'pickle'
            elif ext in ('.yml', '.yaml'):
                format = 'yaml'
            elif ext == '.npz' and HAS_NUMPY:
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
                
        # Load the data based on the format
        if format == 'json':
            with open(filepath, 'r') as f:
                solver_data = json.load(f)
                
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                solver_data = pickle.load(f)
                
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
                
            with open(filepath, 'r') as f:
                solver_data = yaml.safe_load(f)
                
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy format. Install with: pip install numpy")
                
            # NumPy files contain serialized JSON metadata plus sampled values
            np_data = np.load(filepath)
            solver_data = json.loads(np_data['metadata'].item())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return cls._deserialize(solver_data)
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> 'Solver':
        """Deserialize a solver from a dictionary.
        
        Args:
            data: Dictionary representation of the solver
            
        Returns:
            The deserialized Solver
        """
        # Check version if available
        if "version" in data:
            file_version = data["version"]
            if file_version != SOLVER_FORMAT_VERSION:
                print(f"Warning: Solver file version ({file_version}) does not match current version ({SOLVER_FORMAT_VERSION}). Some features may not work correctly.")
        
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
            
            # Process each channel
            for channel_name, channel_data in spline_data.items():
                # Create a new channel
                interpolation = channel_data.get("interpolation", "cubic")
                min_max = channel_data.get("min-max")
                
                channel = spline.add_channel(
                    name=channel_name,
                    interpolation=interpolation,
                    min_max=min_max
                )
                
                # Add keyframes
                for kf_data in channel_data.get("keyframes", []):
                    at = kf_data["@"]
                    value = kf_data["value"]
                    interpolation = kf_data.get("interpolation")
                    control_points = kf_data.get("control-points")
                    derivative = kf_data.get("derivative")
                    
                    channel.add_keyframe(
                        at=at,
                        value=value,
                        interpolation=interpolation,
                        control_points=control_points,
                        derivative=derivative
                    )
        
        return solver