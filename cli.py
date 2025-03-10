#!/usr/bin/env python3
"""Command-line interface for the splinaltap library."""

import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

from .interpolator import KeyframeInterpolator
from .visualization import plot_interpolation_comparison, plot_single_interpolation

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
    
    return parser


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
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())