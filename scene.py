"""
Scene management for multiple keyframe interpolations.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

from .interpolator import KeyframeInterpolator

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


class Scene:
    """A scene containing multiple keyframe interpolators."""
    
    def __init__(self, name: str = "Untitled"):
        """Initialize a new scene.
        
        Args:
            name: The name of the scene
        """
        self.name = name
        self.interpolators: Dict[str, KeyframeInterpolator] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_interpolator(self, name: str, interpolator: KeyframeInterpolator) -> None:
        """Add an interpolator to the scene.
        
        Args:
            name: The name to identify this interpolator
            interpolator: The KeyframeInterpolator instance to add
        """
        self.interpolators[name] = interpolator
    
    def remove_interpolator(self, name: str) -> None:
        """Remove an interpolator from the scene.
        
        Args:
            name: The name of the interpolator to remove
        """
        if name in self.interpolators:
            del self.interpolators[name]
        else:
            raise KeyError(f"No interpolator named '{name}' in scene")
    
    def get_interpolator(self, name: str) -> KeyframeInterpolator:
        """Get an interpolator by name.
        
        Args:
            name: The name of the interpolator to get
            
        Returns:
            The KeyframeInterpolator instance
        """
        if name in self.interpolators:
            return self.interpolators[name]
        else:
            raise KeyError(f"No interpolator named '{name}' in scene")
    
    def get_interpolator_names(self) -> List[str]:
        """Get the names of all interpolators in the scene.
        
        Returns:
            List of interpolator names
        """
        return list(self.interpolators.keys())
    
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
    
    def _serialize_interpolator(self, interpolator: KeyframeInterpolator) -> Dict[str, Any]:
        """Serialize an interpolator to a dictionary.
        
        Args:
            interpolator: The interpolator to serialize
            
        Returns:
            Dictionary representation of the interpolator
        """
        # Start with basic information
        data = {
            "num_indices": interpolator.num_indices,
            "time_range": interpolator.time_range,
            "keyframes": []
        }
        
        # Add keyframes
        for index, (_, derivative, control_points) in sorted(interpolator.keyframes.items()):
            kf_data = {"index": index}
            
            # Try to get the value or expression
            value = interpolator._evaluate_keyframe(index, index)
            kf_data["value"] = value
            
            if derivative is not None:
                kf_data["derivative"] = derivative
                
            if control_points is not None:
                kf_data["control_points"] = list(control_points)
                
            data["keyframes"].append(kf_data)
            
        return data
    
    def save(self, filepath: str, format: Optional[str] = None) -> None:
        """Save the scene to a file.
        
        Args:
            filepath: The path to save to
            format: Optional format override, one of ('json', 'pickle', 'python', 'yaml', 'numpy')
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
            elif ext == '.py':
                format = 'python'
            elif ext in ('.yml', '.yaml'):
                format = 'yaml'
            elif ext == '.npz' and HAS_NUMPY:
                format = 'numpy'
            else:
                format = 'json'  # Default to JSON
                
        # Prepare the scene data
        scene_data = {
            "name": self.name,
            "metadata": self.metadata,
            "interpolators": {}
        }
        
        for name, interpolator in self.interpolators.items():
            scene_data["interpolators"][name] = self._serialize_interpolator(interpolator)
            
        # Save in the requested format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(scene_data, f, indent=2)
                
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(scene_data, f)
                
        elif format == 'python':
            with open(filepath, 'w') as f:
                f.write("# splinaltap scene file - generated code\n\n")
                f.write("from splinaltap import KeyframeInterpolator, Scene\n\n")
                f.write(f"def create_scene():\n")
                f.write(f"    scene = Scene(name=\"{self.name}\")\n\n")
                
                # Add metadata
                if self.metadata:
                    f.write("    # Set metadata\n")
                    for key, value in self.metadata.items():
                        f.write(f"    scene.set_metadata(\"{key}\", {repr(value)})\n")
                    f.write("\n")
                
                # Create each interpolator
                for name, interpolator in self.interpolators.items():
                    # Add interpolator creation
                    f.write(f"    # Create interpolator: {name}\n")
                    f.write(f"    {name} = KeyframeInterpolator(")
                    if interpolator.num_indices is not None:
                        f.write(f"num_indices={interpolator.num_indices}")
                    if interpolator.time_range is not None:
                        f.write(f", time_range={interpolator.time_range}")
                    f.write(")\n")
                    
                    # Add keyframes
                    for index, (_, derivative, control_points) in sorted(interpolator.keyframes.items()):
                        value = interpolator._evaluate_keyframe(index, index)
                        f.write(f"    {name}.set_keyframe({index}, {repr(value)}")
                        if derivative is not None or control_points is not None:
                            f.write(f", derivative={repr(derivative)}")
                        if control_points is not None:
                            f.write(f", control_points={repr(control_points)}")
                        f.write(")\n")
                    
                    # Add to scene
                    f.write(f"    scene.add_interpolator(\"{name}\", {name})\n\n")
                
                f.write("    return scene\n\n")
                f.write("if __name__ == \"__main__\":\n")
                f.write("    scene = create_scene()\n")
                
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
                
            with open(filepath, 'w') as f:
                yaml.dump(scene_data, f, sort_keys=False)
                
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy format. Install with: pip install numpy")
                
            # Convert scene data to a format suitable for NumPy
            np_data = {
                'metadata': json.dumps(scene_data)
            }
            
            # For each interpolator, also save sample values for fast loading
            for name, interpolator in self.interpolators.items():
                try:
                    # Get time range
                    t_min, t_max = interpolator.get_time_range()
                    # Sample 1000 points for each interpolator
                    np_data[f"{name}_values"] = interpolator.sample_range(t_min, t_max, 1000, "cubic")
                    np_data[f"{name}_times"] = np.linspace(t_min, t_max, 1000)
                except Exception as e:
                    warnings.warn(f"Could not sample values for {name}: {e}")
            
            np.savez(filepath, **np_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str, format: Optional[str] = None) -> 'Scene':
        """Load a scene from a file.
        
        Args:
            filepath: The path to load from
            format: Optional format override, one of ('json', 'pickle', 'yaml', 'numpy')
                   If not specified, inferred from the file extension
                   
        Returns:
            The loaded Scene
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
                scene_data = json.load(f)
                
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                scene_data = pickle.load(f)
                
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
                
            with open(filepath, 'r') as f:
                scene_data = yaml.safe_load(f)
                
        elif format == 'numpy':
            if not HAS_NUMPY:
                raise ImportError("NumPy is required for NumPy format. Install with: pip install numpy")
                
            # NumPy files contain serialized JSON metadata plus sampled values
            np_data = np.load(filepath)
            scene_data = json.loads(np_data['metadata'].item())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Create and populate the scene
        scene = cls(name=scene_data.get("name", "Untitled"))
        
        # Add metadata
        scene.metadata = scene_data.get("metadata", {})
        
        # Create interpolators
        for name, interp_data in scene_data.get("interpolators", {}).items():
            num_indices = interp_data.get("num_indices")
            time_range = interp_data.get("time_range")
            
            interpolator = KeyframeInterpolator(num_indices, time_range)
            
            # Add keyframes
            for kf_data in interp_data.get("keyframes", []):
                index = kf_data["index"]
                value = kf_data["value"]
                derivative = kf_data.get("derivative")
                control_points = tuple(kf_data["control_points"]) if "control_points" in kf_data else None
                
                interpolator.set_keyframe(index, value, derivative, control_points)
                
            scene.add_interpolator(name, interpolator)
            
        return scene