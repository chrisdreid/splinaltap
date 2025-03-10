#!/usr/bin/env python3
"""Command-line interface for the splinaltap library."""

import sys
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

from .interpolator import KeyframeInterpolator
from .visualization import plot_interpolation_comparison, plot_single_interpolation
from .scene import Scene
from .backends import BackendManager

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def create_keyframe_interpolator_from_json(data: Dict) -> KeyframeInterpolator:
    """Create a KeyframeInterpolator from JSON data."""
    # Initialize interpolator
    num_indices = data.get("num_indices")
    time_range = data.get("time_range")
    interpolator = KeyframeInterpolator(num_indices, time_range)
    
    # Add variables
    for name, value in data.get("variables", {}).items():
        interpolator.set_variable(name, value)
    
    # Add keyframes
    for kf_data in data.get("keyframes", []):
        index = kf_data["index"]
        value = kf_data["value"]
        derivative = kf_data.get("derivative")
        control_points = tuple(kf_data["control_points"]) if "control_points" in kf_data else None
        interpolator.set_keyframe(index, value, derivative, control_points)
    
    return interpolator


def save_interpolator_to_json(interpolator: KeyframeInterpolator, filepath: str) -> None:
    """Save interpolator settings to a JSON file."""
    data = {
        "num_indices": interpolator.num_indices,
        "time_range": interpolator.time_range,
        "keyframes": [],
        "variables": {}
    }
    
    # Can't directly serialize lambda functions, so we need to handle variables specially
    # For now, we just skip them since they're not easy to serialize
    
    # Add keyframes - note that we can only save the index and value expression, not derivatives or control points
    for index, (_, derivative, control_points) in sorted(interpolator.keyframes.items()):
        kf_data = {"index": index}
        
        # Get the raw value at this index to save
        value = interpolator._evaluate_keyframe(index, index)
        kf_data["value"] = value
        
        if derivative is not None:
            kf_data["derivative"] = derivative
            
        if control_points is not None:
            kf_data["control_points"] = list(control_points)
            
        data["keyframes"].append(kf_data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def visualize_cmd(args: argparse.Namespace) -> None:
    """Handle the visualize command."""
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    interpolator = create_keyframe_interpolator_from_json(data)
    
    # Create time values to evaluate
    if args.time_range:
        t_min, t_max = map(float, args.time_range.split(','))
    else:
        t_min, t_max = interpolator.get_time_range()
    
    t_values = [t_min + i * (t_max - t_min) / (args.samples - 1) for i in range(args.samples)]
    
    # Visualize
    if args.compare:
        fig = plot_interpolation_comparison(interpolator, t_values, 
                                          methods=args.methods.split(',') if args.methods else None,
                                          title=args.title)
    else:
        fig = plot_single_interpolation(interpolator, t_values, 
                                       method=args.method, 
                                       title=args.title)
    
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Handle the evaluate command."""
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    interpolator = create_keyframe_interpolator_from_json(data)
    
    # Parse channels if provided
    channels = {}
    if args.channels:
        for channel_str in args.channels.split(','):
            name, value = channel_str.split('=')
            channels[name.strip()] = float(value.strip())
    
    # Evaluate at specified time
    result = interpolator.get_value(args.time, args.method, channels)
    
    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(result))
    else:
        print(result)


def sample_cmd(args: argparse.Namespace) -> None:
    """Handle the sample command."""
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    interpolator = create_keyframe_interpolator_from_json(data)
    
    # Parse channels if provided
    channels = {}
    if args.channels:
        for channel_str in args.channels.split(','):
            name, value = channel_str.split('=')
            channels[name.strip()] = float(value.strip())
    
    # Get time range
    if args.time_range:
        t_min, t_max = map(float, args.time_range.split(','))
    else:
        t_min, t_max = interpolator.get_time_range()
    
    # Sample values
    values = interpolator.sample_range(t_min, t_max, args.samples, args.method, channels)
    
    # Output result
    if args.output:
        if args.output.endswith('.json'):
            with open(args.output, 'w') as f:
                json.dump(values, f)
        elif args.output.endswith('.csv'):
            with open(args.output, 'w') as f:
                for i, value in enumerate(values):
                    t = t_min + i * (t_max - t_min) / (args.samples - 1)
                    f.write(f"{t},{value}\n")
        elif args.output.endswith('.npy') and HAS_NUMPY:
            np.save(args.output, np.array(values))
        else:
            with open(args.output, 'w') as f:
                for value in values:
                    f.write(f"{value}\n")
    else:
        for value in values:
            print(value)


def scene_info_cmd(args: argparse.Namespace) -> None:
    """Handle the scene-info command."""
    scene = Scene.load(args.input)
    
    print(f"Scene: {scene.name}")
    print(f"Number of interpolators: {len(scene.interpolators)}")
    print("\nMetadata:")
    for key, value in scene.metadata.items():
        print(f"  {key}: {value}")
    
    print("\nInterpolators:")
    for name in scene.get_interpolator_names():
        interpolator = scene.get_interpolator(name)
        num_keyframes = len(interpolator.keyframes)
        try:
            time_range = interpolator.get_time_range()
            time_info = f"time range: {time_range[0]} to {time_range[1]}"
        except:
            time_info = "no keyframes"
            
        print(f"  {name}: {num_keyframes} keyframes, {time_info}")
        
def scene_convert_cmd(args: argparse.Namespace) -> None:
    """Handle the scene-convert command."""
    scene = Scene.load(args.input)
    
    # Get format from output file if not specified
    format = args.format
    if format is None:
        _, ext = os.path.splitext(args.output)
        if ext in ('.json', '.pkl', '.pickle', '.py', '.yml', '.yaml', '.npz'):
            format = None  # Let the save method determine it
        else:
            format = 'json'  # Default
    
    # Save to new format
    scene.save(args.output, format=format)
    print(f"Converted scene '{scene.name}' from {args.input} to {args.output}")

def scene_extract_cmd(args: argparse.Namespace) -> None:
    """Handle the scene-extract command."""
    scene = Scene.load(args.input)
    
    # Check if the interpolator exists
    if args.interpolator not in scene.interpolators:
        print(f"Error: No interpolator named '{args.interpolator}' in the scene")
        return 1
    
    interpolator = scene.get_interpolator(args.interpolator)
    
    # Create a new scene with just this interpolator
    new_scene = Scene(name=f"{scene.name} - {args.interpolator}")
    new_scene.add_interpolator(args.interpolator, interpolator)
    
    # Copy relevant metadata
    if 'description' in scene.metadata:
        new_scene.set_metadata('description', 
                             f"Extracted from {scene.name}: {scene.metadata['description']}")
    
    # Save the new scene
    new_scene.save(args.output)
    print(f"Extracted interpolator '{args.interpolator}' from scene '{scene.name}' to {args.output}")

def backend_cmd(args: argparse.Namespace) -> None:
    """Handle the backend command."""
    if args.list:
        # List available backends
        available = BackendManager.available_backends()
        current = BackendManager.get_backend().name
        
        print("Available backends:")
        for name in available:
            backend = BackendManager.get_backend(name)
            features = []
            if backend.supports_gpu:
                features.append("GPU")
            if backend.supports_autodiff:
                features.append("autodiff")
                
            feature_str = f" ({', '.join(features)})" if features else ""
            current_marker = " [current]" if name == current else ""
            
            print(f"  {name}{feature_str}{current_marker}")
            
    elif args.info:
        # Show info about current backend
        backend = BackendManager.get_backend()
        print(f"Current backend: {backend.name}")
        print(f"GPU support: {'Yes' if backend.supports_gpu else 'No'}")
        print(f"Autodiff support: {'Yes' if backend.supports_autodiff else 'No'}")
        
    elif args.use:
        # Set backend
        try:
            BackendManager.set_backend(args.use)
            print(f"Backend set to {args.use}")
        except Exception as e:
            print(f"Error setting backend: {e}")
            return 1
            
    elif args.best:
        # Use best available backend
        original = BackendManager.get_backend().name
        BackendManager.use_best_available()
        new_backend = BackendManager.get_backend().name
        print(f"Using best available backend: {new_backend} (was {original})")
        
    else:
        # Default to showing current backend
        backend = BackendManager.get_backend()
        print(f"Current backend: {backend.name}")

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Splinaltap - Keyframe interpolation tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize interpolation")
    viz_parser.add_argument("input", help="Input JSON file with keyframe data")
    viz_parser.add_argument("-o", "--output", help="Output image file (if not provided, shows interactive plot)")
    viz_parser.add_argument("-s", "--samples", type=int, default=1000, help="Number of samples to evaluate")
    viz_parser.add_argument("-t", "--title", help="Plot title")
    viz_parser.add_argument("-r", "--time-range", help="Time range to visualize (min,max)")
    viz_parser.add_argument("-c", "--compare", action="store_true", help="Compare different interpolation methods")
    viz_parser.add_argument("-m", "--method", default="cubic", help="Interpolation method to use")
    viz_parser.add_argument("--methods", help="Comma-separated list of methods to compare")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate at a specific time")
    eval_parser.add_argument("input", help="Input JSON file with keyframe data")
    eval_parser.add_argument("time", type=float, help="Time to evaluate at")
    eval_parser.add_argument("-m", "--method", default="cubic", help="Interpolation method to use")
    eval_parser.add_argument("-c", "--channels", help="Channel values as name=value,name=value")
    eval_parser.add_argument("-o", "--output", help="Output file (if not provided, prints to console)")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample values in a range")
    sample_parser.add_argument("input", help="Input JSON file with keyframe data")
    sample_parser.add_argument("-o", "--output", help="Output file (if not provided, prints to console)")
    sample_parser.add_argument("-s", "--samples", type=int, default=100, help="Number of samples to evaluate")
    sample_parser.add_argument("-m", "--method", default="cubic", help="Interpolation method to use")
    sample_parser.add_argument("-c", "--channels", help="Channel values as name=value,name=value")
    sample_parser.add_argument("-r", "--time-range", help="Time range to sample (min,max)")
    sample_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    
    # Scene info command
    scene_info_parser = subparsers.add_parser("scene-info", help="Display information about a scene")
    scene_info_parser.add_argument("input", help="Input scene file")
    
    # Scene convert command
    scene_convert_parser = subparsers.add_parser("scene-convert", help="Convert a scene file to a different format")
    scene_convert_parser.add_argument("input", help="Input scene file")
    scene_convert_parser.add_argument("output", help="Output scene file")
    scene_convert_parser.add_argument("-f", "--format", 
                                   choices=["json", "pickle", "python", "yaml", "numpy"],
                                   help="Output format (if not specified, determined from file extension)")
    
    # Scene extract command
    scene_extract_parser = subparsers.add_parser("scene-extract", 
                                              help="Extract an interpolator from a scene to a new file")
    scene_extract_parser.add_argument("input", help="Input scene file")
    scene_extract_parser.add_argument("interpolator", help="Name of the interpolator to extract")
    scene_extract_parser.add_argument("output", help="Output file for the extracted interpolator")
    
    # Backend command
    backend_parser = subparsers.add_parser("backend", help="Manage compute backends")
    backend_group = backend_parser.add_mutually_exclusive_group()
    backend_group.add_argument("--list", action="store_true", help="List available backends")
    backend_group.add_argument("--info", action="store_true", help="Show info about current backend")
    backend_group.add_argument("--use", help="Set the active backend")
    backend_group.add_argument("--best", action="store_true", help="Use the best available backend")
    
    return parser


def sample_cmd(args: argparse.Namespace) -> None:
    """Handle the sample command."""
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    interpolator = create_keyframe_interpolator_from_json(data)
    
    # Parse channels if provided
    channels = {}
    if args.channels:
        for channel_str in args.channels.split(','):
            name, value = channel_str.split('=')
            channels[name.strip()] = float(value.strip())
    
    # Get time range
    if args.time_range:
        t_min, t_max = map(float, args.time_range.split(','))
    else:
        t_min, t_max = interpolator.get_time_range()
    
    # Sample values, with GPU acceleration if requested
    if args.gpu:
        try:
            values = interpolator.sample_with_gpu(t_min, t_max, args.samples, args.method, channels)
            print("Using GPU acceleration")
        except Exception as e:
            print(f"GPU acceleration failed: {e}, falling back to CPU")
            values = interpolator.sample_range(t_min, t_max, args.samples, args.method, channels)
    else:
        values = interpolator.sample_range(t_min, t_max, args.samples, args.method, channels)
    
    # Output result
    if args.output:
        if args.output.endswith('.json'):
            with open(args.output, 'w') as f:
                json.dump(BackendManager.to_numpy(values).tolist(), f)
        elif args.output.endswith('.csv'):
            with open(args.output, 'w') as f:
                values_np = BackendManager.to_numpy(values)
                for i, value in enumerate(values_np):
                    t = t_min + i * (t_max - t_min) / (args.samples - 1)
                    f.write(f"{t},{value}\n")
        elif args.output.endswith('.npy') and HAS_NUMPY:
            values_np = BackendManager.to_numpy(values)
            np.save(args.output, values_np)
        else:
            with open(args.output, 'w') as f:
                values_np = BackendManager.to_numpy(values)
                for value in values_np:
                    f.write(f"{value}\n")
    else:
        values_np = BackendManager.to_numpy(values)
        for value in values_np:
            print(value)

def main():
    """Main entry point for the command-line interface."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "visualize":
        visualize_cmd(args)
    elif args.command == "evaluate":
        evaluate_cmd(args)
    elif args.command == "sample":
        sample_cmd(args)
    elif args.command == "scene-info":
        scene_info_cmd(args)
    elif args.command == "scene-convert":
        scene_convert_cmd(args)
    elif args.command == "scene-extract":
        scene_extract_cmd(args)
    elif args.command == "backend":
        backend_cmd(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())