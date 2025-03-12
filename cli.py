#!/usr/bin/env python3
"""Command-line interface for the splinaltap library."""

import sys
import argparse
import json
import os
import csv
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt

# Use absolute or relative imports based on context
try:
    # When installed as a package
    from splinaltap.interpolator import KeyframeInterpolator
    from splinaltap.visualization import plot_interpolation_comparison, plot_single_interpolation
    from splinaltap.scene import Scene
    from splinaltap.backends import BackendManager
except ImportError:
    # When run directly (python splinaltap)
    from interpolator import KeyframeInterpolator
    from visualization import plot_interpolation_comparison, plot_single_interpolation
    from scene import Scene
    from backends import BackendManager

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
    dimensions = data.get("dimensions", 1)
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


def sanitize_for_ast(value: str) -> str:
    """Sanitize strings for AST parsing by replacing potentially unsafe characters.
    
    Args:
        value: The string value to sanitize
        
    Returns:
        The sanitized string
    """
    # Handle ^ power operator to use pow() function
    if '^' in value:
        import re
        # Replace any occurrence of ^ with ** for Python's power operator
        # This is a more general solution that can handle complex expressions
        value = value.replace('^', '**')
    
    # Replace @ with # which is safer for AST parsing
    return value.replace('@', '#')

def parse_method_parameters(method_str: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """Parse a method specification with optional parameters.
    
    Args:
        method_str: String in format "method{param1=value1,param2=value2}" or just "method"
        
    Returns:
        Tuple of (method_name, parameters_dict)
    """
    if '{' not in method_str:
        return method_str, None
        
    # Split at the first { character
    parts = method_str.split('{', 1)
    method_name = parts[0]
    
    # Extract parameter string and remove trailing }
    param_str = parts[1]
    if param_str.endswith('}'):
        param_str = param_str[:-1]
    
    # Parse parameters
    params = {}
    if param_str:
        # First, handle key=value pairs
        if '=' in param_str:
            # Find the key name before the first =
            key_end = param_str.find('=')
            key = param_str[:key_end].strip()
            # Everything after the key= is the value, might contain commas
            value = param_str[key_end + 1:].strip()
            params[key] = value
        else:
            # No key=value format, just split by commas
            param_parts = param_str.split(',')
            for i, part in enumerate(param_parts):
                params[f"param{i}"] = part.strip()
    
    return method_name, params

def create_keyframe_interpolator_from_args(args: argparse.Namespace) -> KeyframeInterpolator:
    """Create a KeyframeInterpolator from command-line arguments."""
    # If an input file is provided, load from that
    if args.input_file:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        return create_keyframe_interpolator_from_json(data)
    
    # Otherwise, create from keyframes arguments
    # Parse keyframes in format "index:value" or more complex formats
    keyframes = []
    # Check if we're in index mode (for non-normalized keyframes)
    use_index_mode = args.use_indices if hasattr(args, 'use_indices') else False
    
    for kf_str in args.keyframes:
        # Support different formats:
        # - Basic: "0.5:10" (position 0.5, value 10)
        # - With method: "0.5:10@cubic" (position 0.5, value 10, method cubic)
        # - With method parameters: "0.5:10@hermite{deriv=2}" (position 0.5, value 10, hermite method with derivative 2)
        # - With method parameters: "0.5:10@bezier{cp=0.1,0,0.2,3}" (position 0.5, value 10, bezier with control points)
        # - Expression: "0.5:sin(t)@cubic" (position 0.5, expression sin(t), method cubic)
        
        # First, split on @ to separate the position:value part from the method part
        if '@' in kf_str:
            main_part, method_part = kf_str.split('@', 1)
            method_name, params = parse_method_parameters(method_part)
        else:
            main_part = kf_str
            method_name, params = None, None
        
        # Now parse the position:value part
        parts = main_part.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid keyframe format: {kf_str}. Use 'position:value[@method{{parameters}}]' format.")
        
        position = float(parts[0])
        value = parts[1]
        
        # Handle method-specific parameters
        derivative = None
        control_points = None
        
        if method_name == "hermite" and params and "deriv" in params:
            try:
                derivative = float(params["deriv"])
            except ValueError:
                raise ValueError(f"Invalid derivative value in {kf_str}")
        elif method_name == "bezier" and params:
            try:
                # There are two possible formats:
                # 1. bezier{cp=x1,y1,x2,y2} - where cp is the parameter name
                # 2. bezier{x1,y1,x2,y2} - where the values are directly given
                
                cp_values = None
                
                # Check if we have a 'cp' parameter
                if "cp" in params:
                    cp_str = params["cp"].strip()
                    cp_values = [p.strip() for p in cp_str.split(',')]
                # Check if we have direct parameters
                elif len(params) >= 4:
                    # Get values from param0, param1, etc.
                    cp_values = []
                    for i in range(4):
                        if f"param{i}" in params:
                            cp_values.append(params[f"param{i}"])
                
                if cp_values and len(cp_values) == 4:
                    # Convert each part to a float, handling negative numbers
                    control_points = tuple(float(cp) for cp in cp_values)
                else:
                    # If we reach here, the format wasn't recognized
                    raise ValueError(f"Bezier control points must be 4 comma-separated values in {kf_str}")
            except ValueError as e:
                if "Bezier control points" in str(e):
                    raise
                raise ValueError(f"Invalid control point values in {kf_str}")
        
        # Sanitize the value if it's a string expression
        if not isinstance(value, (int, float)):
            try:
                # Try to convert to float first
                value = float(value)
            except ValueError:
                # It's an expression, sanitize it
                value = sanitize_for_ast(value)
        
        # Store keyframe data for later
        keyframes.append((position, value, derivative, control_points))
    
    # Initialize interpolator with the time range (0-1 by default, or None for index mode)
    time_range = None if use_index_mode else (0.0, 1.0)
    interpolator = KeyframeInterpolator(time_range=time_range)
    
    # Add variables if provided
    if args.variables:
        for var_str in args.variables.split(','):
            name, value = var_str.split('=')
            name = name.strip()
            value = value.strip()
            
            # Sanitize value if it's an expression
            try:
                value = float(value)
            except ValueError:
                value = sanitize_for_ast(value)
                
            interpolator.set_variable(name, value)
    
    # Add all keyframes
    for position, value, derivative, control_points in keyframes:
        interpolator.set_keyframe(position, value, derivative, control_points)
    
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
    # Create interpolator from file or direct keyframes
    interpolator = create_keyframe_interpolator_from_args(args)
    
    # Parse sample points
    x_values = []
    channel_methods = {}  # Dict mapping channel names to methods
    
    if args.samples:
        # Check if first value is an integer sample count
        try:
            sample_count = int(args.samples[0])
            # It's a sample count, generate evenly spaced points
            if args.range:
                x_min, x_max = map(float, args.range.split(','))
            else:
                x_min, x_max = interpolator.get_time_range()
                
            x_values = [x_min + i * (x_max - x_min) / (sample_count - 1) for i in range(sample_count)]
            
            # Process any channel:method specs for the entire range
            if len(args.samples) > 1:
                for spec in args.samples[1:]:
                    if '@' in spec:
                        for part in spec.split('@')[1:]:
                            if ':' in part:
                                channel, *methods = part.split(':')
                                channel_methods[channel] = methods
                            else:
                                channel_methods[part] = args.methods
        except ValueError:
            # Not a sample count, process each sample spec
            for spec in args.samples:
                if '@' in spec:
                    # Parse sample with channel:method specs
                    sample_value, specs = parse_sample_spec(spec)
                    x_values.append(sample_value)
                    
                    # Update channel methods
                    for channel, methods in specs.items():
                        if channel not in channel_methods:
                            channel_methods[channel] = []
                        channel_methods[channel].extend(methods)
                else:
                    # Just a plain sample value
                    x_values.append(float(spec))
    else:
        # Default: 100 evenly spaced samples
        sample_count = 100
        if args.range:
            x_min, x_max = map(float, args.range.split(','))
        else:
            x_min, x_max = interpolator.get_time_range()
            
        x_values = [x_min + i * (x_max - x_min) / (sample_count - 1) for i in range(sample_count)]
    
    # Determine methods to visualize
    methods_to_use = args.methods
    if args.compare:
        if channel_methods and "default" in channel_methods:
            methods_to_use = channel_methods["default"]
        # Otherwise, use methods from args
    
    # Visualize
    if args.compare:
        fig = plot_interpolation_comparison(interpolator, x_values, 
                                          methods=methods_to_use,
                                          title=args.title)
    else:
        # Determine which method to use - if channel specified, use that one
        method = args.methods[0]  # Default to first specified method
        if channel_methods and "default" in channel_methods and channel_methods["default"]:
            method = channel_methods["default"][0]  # Use first method
            
        fig = plot_single_interpolation(interpolator, x_values, 
                                       method=method, 
                                       title=args.title)
    
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Handle the evaluate command."""
    # Create interpolator from file or direct keyframes
    interpolator = create_keyframe_interpolator_from_args(args)
    
    # Parse channels if provided
    channels = {}
    if args.channels:
        for channel_str in args.channels.split(','):
            name, value = channel_str.split('=')
            channels[name.strip()] = float(value.strip())
    
    # Parse samples
    samples = []
    channel_methods = {}  # Dict mapping channel names to methods
    
    if args.samples:
        # Check if first value is an integer sample count
        try:
            sample_count = int(args.samples[0])
            # It's a sample count, generate evenly spaced samples
            if args.range:
                x_min, x_max = map(float, args.range.split(','))
            else:
                x_min, x_max = interpolator.get_time_range()
                
            samples = [x_min + i * (x_max - x_min) / (sample_count - 1) for i in range(sample_count)]
            
            # Process any channel:method specs for the entire range
            if len(args.samples) > 1:
                for spec in args.samples[1:]:
                    if '@' in spec:
                        for part in spec.split('@')[1:]:
                            if ':' in part:
                                channel, *methods = part.split(':')
                                if methods and methods[0]:
                                    channel_methods[channel] = methods[0]  # Just take first method for evaluate
                            else:
                                channel_methods[part] = args.methods[0]
        except ValueError:
            # Not a sample count, process each sample spec
            for spec in args.samples:
                if '@' in spec:
                    # Parse sample with channel:method specs
                    sample_value, specs = parse_sample_spec(spec)
                    samples.append(sample_value)
                    
                    # For evaluate, just take the first method for each channel
                    for channel, methods in specs.items():
                        if methods and methods[0]:
                            channel_methods[channel] = methods[0]
                        else:
                            channel_methods[channel] = args.methods[0]
                else:
                    # Just a plain sample value
                    samples.append(float(spec))
    else:
        # Default if no samples specified: single sample at 0.0
        samples = [0.0]
    
    # Determine which method to use - if channel specified, use that one
    method = args.methods[0]  # Default to first method
    if channel_methods and "default" in channel_methods:
        method = channel_methods["default"]
    
    # Evaluate at all samples
    values = []
    for sample in samples:
        values.append(interpolator.get_value(sample, method, channels))
    
    # Convert to a format compatible with format_output
    results = {
        "default": {
            method: values
        }
    }
    
    # Determine content type from args or file extension
    content_type = args.content_type
    
    if not content_type and args.output_file:
        # Try to determine from file extension
        ext = os.path.splitext(args.output_file)[1].lower()
        if ext == '.json':
            content_type = 'json'
        elif ext == '.csv':
            content_type = 'csv'
        elif ext == '.yaml' or ext == '.yml':
            content_type = 'yaml'
        elif ext == '.txt':
            content_type = 'text'
        elif ext == '.npy' and HAS_NUMPY:
            # Special case for numpy
            if HAS_NUMPY:
                # Save to numpy format
                data = np.column_stack([samples, values])
                np.save(args.output_file, data)
                print(f"Saved results to {args.output_file} in NumPy format")
                return
    
    # Default to json if not specified
    if not content_type:
        content_type = 'json'
    
    # If only a single sample was evaluated and plain text output is requested, simplify
    if len(samples) == 1 and content_type == 'text' and not args.output_file:
        print(values[0])
        return
    
    # Format and output the results
    format_output({}, samples, results, content_type, args.output_file)


def parse_sample_spec(sample_spec: str) -> Tuple[float, Dict[str, List[str]]]:
    """Parse a sample specification in the format VALUE[@CHANNEL:METHOD[@CHANNEL:METHOD...]]
    
    Args:
        sample_spec: A string like "0.5" or "0.5@channel-a:linear@channel-b:cubic:hermite"
        
    Returns:
        A tuple of (sample_value, {channel: [methods]})
    """
    parts = sample_spec.split('@')
    
    # Sample value must be a float
    sample_value = float(parts[0])
        
    # Process channel and method specifications
    channel_methods = {}
    for part in parts[1:]:
        if ':' in part:
            channel, *methods = part.split(':')
            # Sanitize method names if needed
            sanitized_methods = [sanitize_for_ast(m) for m in methods]
            channel_methods[channel] = sanitized_methods
        else:
            # If no methods specified, use default method
            channel_methods[part] = []
            
    return (sample_value, channel_methods)
    
def format_output(data: Dict, samples: List[float], results: Dict, content_type: str = "json", output_file: Optional[str] = None) -> None:
    """Format and output the results based on the specified content type.
    
    Args:
        data: The data to output
        samples: The samples to output
        results: The results dictionary
        content_type: The content type/format (json, csv, text, yaml)
        output_file: Optional output file path
    """
    formatted_output = None
    
    if content_type == "json":
        # Format as JSON
        output = {"samples": samples, "results": {}}
        
        for channel, method_values in results.items():
            output["results"][channel] = {}
            for method, values in method_values.items():
                # Convert to list for safe JSON serialization
                values_as_list = values if isinstance(values, list) else list(values)
                output["results"][channel][method] = values_as_list
        
        formatted_output = json.dumps(output, indent=2)
        
    elif content_type == "csv":
        # Format as CSV
        import io
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        
        # Write header
        header = ["sample"]
        for channel, method_values in results.items():
            for method in method_values.keys():
                header.append(f"{channel}_{method}")
        csv_writer.writerow(header)
        
        # Write data rows
        for i, sample in enumerate(samples):
            row = [sample]
            for channel, method_values in results.items():
                for method, values in method_values.items():
                    # Convert to list to ensure safe access
                    values_as_list = values if isinstance(values, list) else list(values)
                    row.append(values_as_list[i])
            csv_writer.writerow(row)
        
        formatted_output = csv_buffer.getvalue()
        
    elif content_type == "yaml":
        # Format as YAML
        output = {"samples": samples, "results": {}}
        
        for channel, method_values in results.items():
            output["results"][channel] = {}
            for method, values in method_values.items():
                # Convert to list for safe serialization
                values_as_list = values if isinstance(values, list) else list(values)
                output["results"][channel][method] = values_as_list
        
        formatted_output = yaml.dump(output, sort_keys=False)
        
    else:  # Default to "text"
        # Format as plain text
        lines = []
        for i, sample in enumerate(samples):
            line = f"{sample}"
            for channel, method_values in results.items():
                for method, values in method_values.items():
                    # Convert to list to ensure safe access
                    values_as_list = values if isinstance(values, list) else list(values)
                    line += f",{channel}:{method}={values_as_list[i]}"
            lines.append(line)
        
        formatted_output = "\n".join(lines)
    
    # Output the formatted data
    if output_file:
        with open(output_file, 'w') as f:
            f.write(formatted_output)
    else:
        print(formatted_output)

def sample_cmd(args: argparse.Namespace) -> None:
    """Handle the sample command."""
    # Create interpolator from file or direct keyframes
    interpolator = create_keyframe_interpolator_from_args(args)
    
    # Parse channels if provided
    channels = {}
    if args.channels:
        for channel_str in args.channels.split(','):
            name, value = channel_str.split('=')
            channels[name.strip()] = float(value.strip())
    
    # Parse samples
    samples = []
    all_methods = set()
    channel_methods = {}  # Dict mapping channel names to methods
    
    if args.samples:
        # Check if first value is an integer sample count
        try:
            sample_count = int(args.samples[0])
            # It's a sample count, generate evenly spaced samples
            if args.range:
                x_min, x_max = map(float, args.range.split(','))
            else:
                x_min, x_max = interpolator.get_time_range()
                
            samples = [x_min + i * (x_max - x_min) / (sample_count - 1) for i in range(sample_count)]
            
            # Process any channel:method specs for the entire range
            if len(args.samples) > 1:
                for spec in args.samples[1:]:
                    if '@' in spec:
                        for part in spec.split('@')[1:]:
                            if ':' in part:
                                channel, *methods = part.split(':')
                                channel_methods[channel] = methods
                                all_methods.update(methods)
                            else:
                                channel_methods[part] = args.methods
                                all_methods.update(args.methods)
        except ValueError:
            # Not a sample count, process each sample spec
            for spec in args.samples:
                if '@' in spec:
                    # Parse sample with channel:method specs
                    sample_value, specs = parse_sample_spec(spec)
                    samples.append(sample_value)
                    
                    # Update channel methods
                    for channel, methods in specs.items():
                        if channel not in channel_methods:
                            channel_methods[channel] = []
                        channel_methods[channel].extend(methods)
                        all_methods.update(methods)
                else:
                    # Just a plain sample value
                    samples.append(float(spec))
    else:
        # Default if no samples specified: 100 evenly spaced
        sample_count = 100
        if args.range:
            x_min, x_max = map(float, args.range.split(','))
        else:
            x_min, x_max = interpolator.get_time_range()
            
        samples = [x_min + i * (x_max - x_min) / (sample_count - 1) for i in range(sample_count)]
    
    # If no specific methods were parsed from sample specs, use the methods list
    if not all_methods:
        all_methods = set(args.methods)
    
    # Evaluate for each sample position, method, and channel combination
    results = {}
    
    # If no channel methods specified, use all methods for the default channel
    if not channel_methods and samples:
        # Base case: just normal sampling with methods
        values_by_method = {}
        
        # For each method, evaluate at all samples
        for method in all_methods:
            if len(samples) >= 10 and args.use_gpu:
                # For many samples, use optimized sampling
                try:
                    # Need to determine range and count
                    x_min, x_max = min(samples), max(samples)
                    if len(set(samples)) == len(samples) and len(samples) > 1:
                        # Check if evenly spaced
                        diffs = [samples[i+1] - samples[i] for i in range(len(samples)-1)]
                        if max(diffs) - min(diffs) < 1e-10:  # Approximately equal spacing
                            # Use GPU for evenly spaced samples
                            values = interpolator.sample_with_gpu(x_min, x_max, len(samples), method, channels)
                            values_by_method[method] = values
                            print(f"Using GPU acceleration for method: {method}")
                            continue
                except Exception as e:
                    print(f"GPU acceleration failed: {e}, falling back to CPU")
            
            # Standard sample-by-sample evaluation
            values = []
            for sample in samples:
                values.append(interpolator.get_value(sample, method, channels))
            values_by_method[method] = values
        
        results["default"] = values_by_method
    else:
        # Process each channel with its methods
        for channel, methods in channel_methods.items():
            channel_channels = channels.copy()
            values_by_method = {}
            
            # For each method, evaluate at all samples
            actual_methods = methods if methods else list(all_methods)
            for method in actual_methods:
                values = []
                for sample in samples:
                    values.append(interpolator.get_value(sample, method, channel_channels))
                values_by_method[method] = values
            
            results[channel] = values_by_method
    
    # Determine content type from args or file extension
    content_type = args.content_type
    
    if not content_type and args.output_file:
        # Try to determine from file extension
        ext = os.path.splitext(args.output_file)[1].lower()
        if ext == '.json':
            content_type = 'json'
        elif ext == '.csv':
            content_type = 'csv'
        elif ext == '.yaml' or ext == '.yml':
            content_type = 'yaml'
        elif ext == '.txt':
            content_type = 'text'
        elif ext == '.npy' and HAS_NUMPY:
            # Special case for numpy
            # Flatten results into columns
            columns = [samples]
            column_names = ['sample']
            
            for channel, method_values in results.items():
                for method, values in method_values.items():
                    # Convert to list for consistent handling
                    values_as_list = values if isinstance(values, list) else list(values)
                    columns.append(values_as_list)
                    column_names.append(f"{channel}_{method}")
            
            # Stack columns and save
            result = np.column_stack(columns)
            np.save(args.output_file, result)
            print(f"Saved results to {args.output_file} in NumPy format")
            return
    
    # Default to json if not specified
    if not content_type:
        content_type = 'json'
    
    # Format and output the results
    format_output({}, samples, results, content_type, args.output_file)


def scene_info_cmd(args: argparse.Namespace) -> None:
    """Handle the scene-info command."""
    scene = Scene.load(args.input_file)
    
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
    scene = Scene.load(args.input_file)
    
    # Get format from output file if not specified
    format = args.format
    if format is None:
        _, ext = os.path.splitext(args.output_file)
        if ext in ('.json', '.pkl', '.pickle', '.py', '.yml', '.yaml', '.npz'):
            format = None  # Let the save method determine it
        else:
            format = 'json'  # Default
    
    # Save to new format
    scene.save(args.output_file, format=format)
    print(f"Converted scene '{scene.name}' from {args.input_file} to {args.output_file}")

def scene_extract_cmd(args: argparse.Namespace) -> None:
    """Handle the scene-extract command."""
    scene = Scene.load(args.input_file)
    
    # Check if the interpolator exists
    if args.interpolator_name not in scene.interpolators:
        print(f"Error: No interpolator named '{args.interpolator_name}' in the scene")
        return 1
    
    interpolator = scene.get_interpolator(args.interpolator_name)
    
    # Create a new scene with just this interpolator
    new_scene = Scene(name=f"{scene.name} - {args.interpolator_name}")
    new_scene.add_interpolator(args.interpolator_name, interpolator)
    
    # Copy relevant metadata
    if 'description' in scene.metadata:
        new_scene.set_metadata('description', 
                             f"Extracted from {scene.name}: {scene.metadata['description']}")
    
    # Save the new scene
    new_scene.save(args.output_file)
    print(f"Extracted interpolator '{args.interpolator_name}' from scene '{scene.name}' to {args.output_file}")
    
def generate_template_cmd(args: argparse.Namespace) -> None:
    """Handle the generate-template command."""
    # Initialize template structure
    template = None
    
    # If an input file was provided, use it as the base
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                template = json.load(f)
                print(f"Loaded template base from {args.input_file}")
        except Exception as e:
            print(f"Warning: Could not load input file: {e}")
            
    # If no template loaded from file, create a new one
    if template is None:
        template = {
            "range": [0.0, 1.0],
            "variables": {
                "amplitude": 2.5,
                "frequency": 0.5
            },
            "keyframes": []
        }
    
    # If keyframes were specified, add them to the template
    if args.keyframes:
        try:
            # Create an interpolator from the keyframes
            interpolator = create_keyframe_interpolator_from_args(args)
            
            # Extract keyframes and add them to the template
            keyframes_list = []
            for position, (value, derivative, control_points) in sorted(interpolator.keyframes.items()):
                # For expressions, store the original expression if possible or evaluate
                if callable(value):
                    # Try to get original expression string from interpolator
                    try:
                        if hasattr(interpolator, '_expressions') and position in interpolator._expressions:
                            value = interpolator._expressions[position]
                        else:
                            value = value(position, position, {})
                    except:
                        value = f"Expression at position {position}"
                
                kf = {
                    "index": position,
                    "value": value
                }
                
                if derivative is not None:
                    kf["derivative"] = derivative
                    
                if control_points is not None:
                    kf["control_points"] = list(control_points)
                    
                keyframes_list.append(kf)
            
            # Update template with keyframes - handle both direct keyframes and dimensions
            if "dimensions" in template:
                # Multi-dimensional template - add to first dimension if possible
                dims = list(template["dimensions"].keys())
                if dims:
                    template["dimensions"][dims[0]]["keyframes"] = keyframes_list
            else:
                # Single-dimensional template
                template["keyframes"] = keyframes_list
                
        except Exception as e:
            print(f"Warning: Could not process keyframes: {e}")
            # Add default keyframes if none exist yet
            if not template.get("keyframes") and "dimensions" not in template:
                template["keyframes"] = [
                    {"index": 0.0, "value": 0},
                    {"index": 0.5, "value": "sin(t * frequency) * amplitude"},
                    {"index": 1.0, "value": 10}
                ]
    elif not args.input_file:
        # No input file and no keyframes, use default keyframes
        template["keyframes"] = [
            {"index": 0.0, "value": 0},
            {"index": 0.5, "value": "sin(t * frequency) * amplitude"},
            {"index": 1.0, "value": 10}
        ]
    
    # If dimensions are specified, create multi-dimensional template
    if args.dimensions and args.dimensions > 1 and "dimensions" not in template:
        # Convert to multi-dimensional if it isn't already
        keyframes = template.get("keyframes", [])
        
        multi_template = {
            "range": template.get("range", [0.0, 1.0]),
            "variables": template.get("variables", {}),
            "dimensions": {}
        }
        
        # Create a dimension for each axis
        dim_names = ["x", "y", "z"] + [f"dim{i}" for i in range(3, args.dimensions)]
        for i, name in enumerate(dim_names[:args.dimensions]):
            # Vary the example keyframes slightly for each dimension
            if i == 0:
                kf_values = keyframes
            else:
                # Create varied keyframes for other dimensions
                kf_values = []
                for kf in keyframes:
                    new_kf = kf.copy()
                    if isinstance(new_kf.get("value"), (int, float)):
                        new_kf["value"] = new_kf["value"] + i * 5  # Offset by 5 * dimension index
                    kf_values.append(new_kf)
            
            multi_template["dimensions"][name] = {
                "keyframes": kf_values
            }
        
        template = multi_template
    
    # Add scene wrapper if requested
    if args.scene and "interpolators" not in template:
        scene_template = {
            "name": "MyAnimation",
            "metadata": {
                "description": "Generated template animation",
                "author": "SplinalTap User",
                "created": "Generated template"
            },
            "variables": template.get("variables", {}),
            "interpolators": {
                "main": template
            }
        }
        template = scene_template
    
    # Save the template to the output file
    content_type = args.content_type
    
    if not args.output_file:
        print("Error: --output-file is required for --generate-template")
        return 1
    
    # If no content type specified, determine from output file extension
    if not content_type and args.output_file:
        ext = os.path.splitext(args.output_file)[1].lower()
        if ext == '.yaml' or ext == '.yml':
            content_type = 'yaml'
        else:
            content_type = 'json'
    
    with open(args.output_file, 'w') as f:
        if content_type == "yaml":
            yaml.dump(template, f, sort_keys=False)
        else:
            # Default to JSON
            json.dump(template, f, indent=2)
            
    print(f"Generated template file: {args.output_file}")

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
    parser = argparse.ArgumentParser(
        description="Splinaltap - Keyframe interpolation and expression evaluation that goes to eleven!",
        epilog="For more information, visit: https://github.com/chrisdreid/splinaltap"
    )
    
    # We'll use the standard --help option, no need for additional help flags
    
    # Create a non-required command argument that will be specified by subcommands
    parser.add_argument("--command", dest="command", help=argparse.SUPPRESS)
    
    # Create a group for the commands to ensure they're mutually exclusive
    command_group = parser.add_argument_group("commands (optional, default is to evaluate keyframes)")
    command_exclusive = command_group.add_mutually_exclusive_group(required=False)
    
    # Add each command as a flag (evaluation is the default when no command specified)
    command_exclusive.add_argument("--visualize", action="store_const", const="visualize", 
                                 dest="command", help="Visualize interpolation")
    command_exclusive.add_argument("--scene-info", action="store_const", const="scene-info", 
                                 dest="command", help="Display information about a scene")
    command_exclusive.add_argument("--scene-convert", action="store_const", const="scene-convert", 
                                 dest="command", help="Convert a scene file to a different format")
    command_exclusive.add_argument("--scene-extract", action="store_const", const="scene-extract", 
                                 dest="command", help="Extract an interpolator from a scene to a new file")
    command_exclusive.add_argument("--backend", action="store_const", const="backend", 
                                 dest="command", help="Manage compute backends")
    command_exclusive.add_argument("--generate-template", action="store_const", const="generate-template",
                                 dest="command", help="Generate a template input file")
    
    # Create arguments for each command
    # Input sources (mutually exclusive group)
    input_group = parser.add_argument_group("input options (use either input-file or keyframes)")
    input_exclusive = input_group.add_mutually_exclusive_group()
    input_exclusive.add_argument("--input-file", help="Input JSON or scene file")
    input_exclusive.add_argument("--keyframes", nargs="+", 
                               help="Define keyframes directly: POSITION:VALUE [POSITION:VALUE...] (e.g. '0:0 0.5:10 1:0')")
    
    parser.add_argument("--output-file", help="Output file (for visualization, sample, or conversion)")
    # Samples parameter with advanced syntax support
    parser.add_argument("--samples", nargs="+", 
                      help="Samples with optional channel and method specifications. Format: VALUE[@CHANNEL:METHOD][@CHANNEL:METHOD...] or sample count (integer)")
    parser.add_argument("--methods", nargs="+", default=["cubic"],
                      help="List of methods to use or compare (space-separated)")
    parser.add_argument("--range", help="X-value range as min,max (e.g. 0,1) when using integer sample count")
    parser.add_argument("--channels", help="Channel values as name=value,name=value")
    parser.add_argument("--variables", help="Variable values as name=value,name=value for keyframe expressions")
    parser.add_argument("--dimensions", type=int, default=1, help="Number of dimensions for the interpolator (default: 1)")
    parser.add_argument("--title", help="Plot title for visualizations")
    parser.add_argument("--compare", action="store_true", help="Compare different interpolation methods")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--use-indices", action="store_true", 
                      help="Use index mode (non-normalized) instead of the default 0-1 normalized range")
    parser.add_argument("--format", choices=["json", "pickle", "python", "yaml", "numpy"],
                      help="Format for scene conversion")
    parser.add_argument("--content-type", choices=["json", "csv", "yaml", "text"],
                      help="Format for output data (default: json or determined from output file extension)")
    parser.add_argument("--scene", action="store_true", help="Generate a scene template (with --generate-template)")
    parser.add_argument("--interpolator-name", help="Name of the interpolator to extract")
    
    # Backend specific options
    parser.add_argument("--list", action="store_true", help="List available backends (with --backend)")
    parser.add_argument("--info", action="store_true", help="Show info about current backend (with --backend)")
    parser.add_argument("--use-backend", dest="use", help="Set the active backend (with --backend)")
    parser.add_argument("--use-best", dest="best", action="store_true", 
                         help="Use the best available backend (with --backend)")
    
    return parser

def main():
    """Main entry point for the command-line interface."""
    parser = create_parser()
    
    try:
        # Parse arguments
        args = parser.parse_args()
        
        # If no command specified, decide whether to evaluate or sample based on parameters
        if not args.command:
            # If samples is an integer or --range is specified, it's a sample operation
            # Otherwise, it's an evaluate operation
            if args.samples and (
                (len(args.samples) > 0 and args.samples[0].isdigit()) or 
                args.range
            ):
                args.command = "sample"
            else:
                args.command = "evaluate"
        
        # The standard --help flag is handled automatically by argparse
        
        # Validate required args based on command
        if args.command == "visualize" and not (args.input_file or args.keyframes):
            print("Error: Either --input-file or --keyframes is required for --visualize")
            return 1
        elif args.command == "evaluate" and not (args.input_file or args.keyframes):
            print("Error: Either --input-file or --keyframes is required")
            return 1
        elif args.command == "scene-info" and not args.input_file:
            print("Error: --input-file is required for --scene-info")
            return 1
        elif args.command == "scene-convert" and (not args.input_file or not args.output_file):
            print("Error: --input-file and --output-file are required for --scene-convert")
            return 1
        elif args.command == "scene-extract" and (not args.input_file or not args.output_file or not args.interpolator_name):
            print("Error: --input-file, --output-file, and --interpolator-name are required for --scene-extract")
            return 1
        elif args.command == "generate-template" and not args.output_file:
            print("Error: --output-file is required for --generate-template")
            return 1
        
        # Execute the appropriate command
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
        elif args.command == "generate-template":
            generate_template_cmd(args)
        elif args.command == "backend":
            backend_cmd(args)
        else:
            # This shouldn't happen, but just in case
            parser.print_help()
            return 1
    except SystemExit as e:
        # Handle argparse's sys.exit behavior when --help is used
        return e.code
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())