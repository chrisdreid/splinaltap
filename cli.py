#!/usr/bin/env python3
"""Command-line interface for the SplinalTap library."""

import sys
import argparse
import json
import os
import csv
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt

# Use absolute or relative imports based on context
try:
    # When installed as a package
    from splinaltap.solver import Solver
    from splinaltap.spline import Spline
    from splinaltap.channel import Channel
    from splinaltap.backends import BackendManager
except ImportError:
    # When run directly (python splinaltap)
    from solver import Solver
    from spline import Spline
    from channel import Channel
    from backends import BackendManager

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


def sanitize_for_ast(value: str) -> str:
    """Sanitize strings for AST parsing by replacing potentially unsafe characters.
    
    Args:
        value: The string value to sanitize
        
    Returns:
        The sanitized string
    """
    # Replace ^ with ** for Python's power operator
    value = value.replace('^', '**')
    
    # @ symbol is now used directly in expressions
    return value


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


def create_solver_from_args(args: argparse.Namespace) -> Solver:
    """Create a Solver from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Solver object with initialized splines and channels
    """
    # If an input file is provided, load from that
    if args.input_file:
        return Solver.load(args.input_file)
    
    # Otherwise, create from keyframes arguments
    solver = Solver(name="CommandLine")
    
    # Parse keyframes in format "@:value" or more complex formats
    keyframes = []
    
    # Check if we're in index mode (for non-normalized keyframes)
    use_index_mode = args.use_indices if hasattr(args, 'use_indices') else False
    
    for kf_str in args.keyframes:
        # Support different formats:
        # - Basic: "0.5:10" (position 0.5, value 10)
        # - With method: "0.5:10@cubic" (position 0.5, value 10, method cubic)
        # - With method parameters: "0.5:10@hermite{deriv=2}" (position 0.5, value 10, hermite method with derivative 2)
        # - With method parameters: "0.5:10@bezier{cp=0.1,0,0.2,3}" (position 0.5, value 10, bezier with control points)
        # - Expression: "0.5:sin(@)@cubic" (position 0.5, expression sin(@), method cubic)
        
        # First, split on @ to separate the position:value part from the method part
        if '@' in kf_str and ':' in kf_str.split('@')[0]:
            main_part, method_part = kf_str.split('@', 1)
            method_name, params = parse_method_parameters(method_part)
        else:
            main_part = kf_str
            method_name, params = None, None
        
        # Now parse the position:value part
        parts = main_part.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid keyframe format: {kf_str}. Use '@:value[@method{{parameters}}]' format.")
        
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
        
        # Store keyframe data for later (including method name)
        keyframes.append((position, value, derivative, control_points, method_name))
    
    # Initialize default spline and channel
    spline = solver.create_spline("default")
    channel = spline.add_channel("value")
    
    # Add variables if provided
    if hasattr(args, 'variables') and args.variables:
        for var_str in args.variables.split(','):
            name, value = var_str.split('=')
            name = name.strip()
            value = value.strip()
            
            # Sanitize value if it's an expression
            try:
                value = float(value)
            except ValueError:
                value = sanitize_for_ast(value)
                
            solver.set_variable(name, value)
    
    # Add all keyframes
    for position, value, derivative, control_points, method in keyframes:
        # If using indices, normalize to 0-1 range
        if use_index_mode:
            # Find the min and max positions
            min_pos = min(pos for pos, _, _, _, _ in keyframes)
            max_pos = max(pos for pos, _, _, _, _ in keyframes)
            range_size = max_pos - min_pos
            
            # Normalize position
            if range_size > 0:
                at = (position - min_pos) / range_size
            else:
                at = 0
        else:
            at = position
        
        # Add keyframe to the channel
        channel.add_keyframe(
            at=at,
            value=value,
            interpolation=method,
            control_points=control_points,
            derivative=derivative
        )
    
    return solver


def visualize_solver(solver: Solver, args: argparse.Namespace) -> None:
    """Visualize the solver with matplotlib.
    
    Args:
        solver: The solver to visualize
        args: Command-line arguments with visualization options
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample points
    if args.samples:
        try:
            # Try to parse as integer sample count
            sample_count = int(args.samples[0])
            
            # Generate evenly spaced sample points
            sample_points = [i / (sample_count - 1) for i in range(sample_count)]
        except ValueError:
            # It's a list of specific sample points
            sample_points = [float(s.split('@')[0]) if '@' in s else float(s) for s in args.samples]
    else:
        # Default to 100 sample points
        sample_count = 100
        sample_points = [i / (sample_count - 1) for i in range(sample_count)]
    
    # Iterate through all splines
    for spline_name, spline in solver.splines.items():
        # Iterate through all channels in the spline
        for channel_name, channel in spline.channels.items():
            # Generate label
            label = f"{spline_name}.{channel_name}"
            
            # Sample and plot values
            values = []
            for at in sample_points:
                values.append(channel.get_value(at))
            
            # Plot the values
            ax.plot(sample_points, values, marker='o', markersize=3, label=label)
            
            # Also plot the keyframe points
            keyframe_points = channel.get_keyframe_values()
            x_kf = [kf[0] for kf in keyframe_points]
            y_kf = [kf[1] for kf in keyframe_points]
            ax.scatter(x_kf, y_kf, color='red', s=50, zorder=5)
    
    # Set up the plot
    ax.set_xlabel('Time (@)')
    ax.set_ylabel('Value')
    ax.set_title('SplinalTap Interpolation')
    ax.grid(True)
    ax.legend()
    
    # Save or display the plot
    if args.output_file:
        plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output_file}")
    else:
        plt.show()


def sample_solver(solver: Solver, args: argparse.Namespace) -> Dict[str, Any]:
    """Sample the solver at specified points.
    
    Args:
        solver: The solver to sample
        args: Command-line arguments with sampling options
        
    Returns:
        Dictionary with sampling results
    """
    # Parse sample points
    if args.samples:
        try:
            # Try to parse as integer sample count
            sample_count = int(args.samples[0])
            
            # Generate evenly spaced sample points
            sample_points = [i / (sample_count - 1) for i in range(sample_count)]
        except ValueError:
            # It's a list of specific sample points
            sample_points = [float(s.split('@')[0]) if '@' in s else float(s) for s in args.samples]
    else:
        # Default to single sample at 0.5
        sample_points = [0.5]
    
    # Initialize results
    results = {
        "version": "2.0",
        "samples": sample_points,
        "results": {}
    }
    
    # Sample the solver at each point
    for at in sample_points:
        # For each spline
        for spline_name, spline in solver.splines.items():
            # For each channel in the spline
            for channel_name, channel in spline.channels.items():
                # Get the full path (spline.channel)
                full_name = f"{spline_name}.{channel_name}"
                
                # Initialize results for this channel if needed
                if full_name not in results["results"]:
                    results["results"][full_name] = []
                
                # Sample and store the value
                value = channel.get_value(at)
                results["results"][full_name].append(value)
    
    return results


def save_results(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Save sampling results to a file.
    
    Args:
        results: The sampling results to save
        args: Command-line arguments with output options
    """
    # Determine output format
    content_type = getattr(args, 'content_type', 'json')
    
    # Prepare output
    if args.output_file:
        output_file = args.output_file
    else:
        # Default filename based on content type
        output_file = f"output.{content_type}"
    
    # Save based on content type
    if content_type == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif content_type == 'csv':
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['@'] + list(results['results'].keys())
            writer.writerow(header)
            
            # Write data rows
            for i, at in enumerate(results['samples']):
                row = [at]
                for channel in results['results'].keys():
                    row.append(results['results'][channel][i])
                writer.writerow(row)
    elif content_type == 'yaml':
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML output. Install with: pip install pyyaml")
        
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    elif content_type == 'text':
        with open(output_file, 'w') as f:
            f.write(f"SplinalTap Sampling Results\n")
            f.write(f"{'@':<10}")
            
            # Write header
            for channel in results['results'].keys():
                f.write(f"{channel:<15}")
            f.write("\n")
            
            # Write data rows
            for i, at in enumerate(results['samples']):
                f.write(f"{at:<10.4f}")
                for channel in results['results'].keys():
                    f.write(f"{results['results'][channel][i]:<15.4f}")
                f.write("\n")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    print(f"Results saved to {output_file}")


def print_results(results: Dict[str, Any]) -> None:
    """Print sampling results to the console.
    
    Args:
        results: The sampling results to print
    """
    # Print header
    print(f"{'@':<10}", end="")
    for channel in results['results'].keys():
        print(f"{channel:<15}", end="")
    print()
    
    # Print data rows
    for i, at in enumerate(results['samples']):
        print(f"{at:<10.4f}", end="")
        for channel in results['results'].keys():
            print(f"{results['results'][channel][i]:<15.4f}", end="")
        print()


def scene_cmd(args: argparse.Namespace) -> None:
    """Handle the scene command.
    
    Args:
        args: Command-line arguments with scene options
    """
    # Get the scene subcommand and arguments
    if not args.scene:
        print("Error: No scene command specified.")
        return
    
    # Parse scene command and args
    scene_args = args.scene.split()
    if not scene_args:
        print("Error: No scene command specified.")
        return
    
    scene_command = scene_args[0]
    scene_args = scene_args[1:]
    
    if scene_command == "info":
        if len(scene_args) != 1:
            print("Error: Scene info command requires a file path.")
            return
        
        file_path = scene_args[0]
        try:
            solver = Solver.load(file_path)
            print(f"Solver: {solver.name}")
            print(f"Metadata: {solver.metadata}")
            print(f"Range: {solver.range}")
            print(f"Variables: {solver.variables}")
            print(f"Splines: {len(solver.splines)}")
            
            for name, spline in solver.splines.items():
                print(f"  - {name}: {len(spline.channels)} channels")
                for channel_name, channel in spline.channels.items():
                    print(f"    - {channel_name}: {len(channel.keyframes)} keyframes, {channel.interpolation} interpolation")
        except Exception as e:
            print(f"Error reading scene file: {e}")
    
    elif scene_command == "ls":
        if len(scene_args) != 1:
            print("Error: Scene ls command requires a file path.")
            return
        
        file_path = scene_args[0]
        try:
            solver = Solver.load(file_path)
            print(f"Solver: {solver.name}")
            
            for name in solver.get_spline_names():
                print(f"  - {name}")
        except Exception as e:
            print(f"Error reading scene file: {e}")
    
    elif scene_command == "convert":
        if len(scene_args) != 2:
            print("Error: Scene convert command requires input and output file paths.")
            return
        
        input_path, output_path = scene_args
        try:
            solver = Solver.load(input_path)
            solver.save(output_path)
            print(f"Converted {input_path} to {output_path}")
        except Exception as e:
            print(f"Error converting scene file: {e}")
    
    elif scene_command == "extract":
        if len(scene_args) not in (3, 4):
            print("Error: Scene extract command requires input file, output file, and spline path.")
            return
        
        if len(scene_args) == 3:
            input_path, output_path, spline_path = scene_args
            channel_path = None
        else:
            input_path, output_path, spline_path, channel_path = scene_args
        
        try:
            solver = Solver.load(input_path)
            
            if '.' in spline_path and channel_path is None:
                # Format is "spline.channel"
                spline_name, channel_name = spline_path.split('.', 1)
                channel_path = channel_name
            else:
                spline_name = spline_path
            
            if spline_name not in solver.splines:
                print(f"Error: Spline '{spline_name}' not found in solver.")
                return
            
            # Extract the spline
            spline = solver.get_spline(spline_name)
            
            if channel_path:
                # Extract a specific channel
                if channel_path not in spline.channels:
                    print(f"Error: Channel '{channel_path}' not found in spline '{spline_name}'.")
                    return
                
                # Create a new solver with just this channel
                new_solver = Solver(name=f"{solver.name}_{spline_name}_{channel_path}")
                new_spline = new_solver.create_spline(spline_name)
                
                # Copy the channel
                channel = spline.channels[channel_path]
                new_channel = new_spline.add_channel(
                    name=channel_path,
                    interpolation=channel.interpolation,
                    min_max=channel.min_max
                )
                
                # Copy keyframes
                for kf in channel.keyframes:
                    new_channel.add_keyframe(
                        at=kf.at,
                        value=kf.value(kf.at, {}),  # Evaluate to get value
                        interpolation=kf.interpolation,
                        control_points=kf.control_points,
                        derivative=kf.derivative
                    )
                
                # Save the new solver
                new_solver.save(output_path)
                print(f"Extracted channel '{spline_name}.{channel_path}' to {output_path}")
            
            else:
                # Extract the whole spline
                new_solver = Solver(name=f"{solver.name}_{spline_name}")
                new_spline = new_solver.create_spline(spline_name)
                
                # Copy all channels
                for channel_name, channel in spline.channels.items():
                    new_channel = new_spline.add_channel(
                        name=channel_name,
                        interpolation=channel.interpolation,
                        min_max=channel.min_max
                    )
                    
                    # Copy keyframes
                    for kf in channel.keyframes:
                        new_channel.add_keyframe(
                            at=kf.at,
                            value=kf.value(kf.at, {}),  # Evaluate to get value
                            interpolation=kf.interpolation,
                            control_points=kf.control_points,
                            derivative=kf.derivative
                        )
                
                # Save the new solver
                new_solver.save(output_path)
                print(f"Extracted spline '{spline_name}' to {output_path}")
        
        except Exception as e:
            print(f"Error extracting from scene file: {e}")
    
    else:
        print(f"Error: Unknown scene command '{scene_command}'.")


def backend_cmd(args: argparse.Namespace) -> bool:
    """Handle the backend command.
    
    Args:
        args: Command-line arguments with backend options
        
    Returns:
        True if the command should exit after handling backends, False to continue
    """
    # Check if --backend is provided
    if not hasattr(args, 'backend') or not args.backend:
        return False
    
    backend_arg = args.backend
    
    # Quick check: does this handle a specific backend (no sub-arguments)
    if backend_arg and ' ' not in backend_arg:
        # Check if it's asking for one of the standard commands
        if backend_arg.lower() in ('ls', 'list', 'info'):
            pass  # Fall through to command handling below
        else:
            # Try to set the backend
            try:
                BackendManager.set_backend(backend_arg)
                print(f"Backend set to {BackendManager.get_backend().name}")
                
                # Check if there are other actions to perform
                has_other_action = (
                    hasattr(args, 'input_file') and args.input_file or
                    hasattr(args, 'keyframes') and args.keyframes or
                    hasattr(args, 'visualize') and args.visualize or
                    hasattr(args, 'scene') and args.scene
                )
                
                if has_other_action:
                    # We're using backend with other commands - don't exit
                    print(f"Running with backend: {BackendManager.get_backend().name}")
                    return False
                else:
                    # Just setting the backend - exit after this
                    return True
            except Exception as e:
                print(f"Error setting backend: {e}")
                return True
    
    # Handle backend commands
    if not backend_arg or backend_arg.lower() == 'ls' or backend_arg.lower() == 'list':
        # List available backends
        backends = BackendManager.available_backends()
        current = BackendManager.get_backend().name
        
        print("Available backends:")
        for name in backends:
            if name == current:
                print(f"* {name} (current)")
            else:
                print(f"  {name}")
        
        return True
    
    elif backend_arg.lower() == 'info':
        # Show current backend info
        backend = BackendManager.get_backend()
        print(f"Current backend: {backend.name}")
        print(f"Supports GPU: {backend.supports_gpu}")
        print(f"Available math functions: {', '.join(backend.get_math_functions().keys())}")
        
        return True
    
    elif backend_arg.lower() == 'best':
        # Use the best available backend
        BackendManager.use_best_available()
        print(f"Selected best available backend: {BackendManager.get_backend().name}")
        
        # Check if there are other actions to perform
        has_other_action = (
            hasattr(args, 'input_file') and args.input_file or
            hasattr(args, 'keyframes') and args.keyframes or
            hasattr(args, 'visualize') and args.visualize or
            hasattr(args, 'scene') and args.scene
        )
        
        if has_other_action:
            # We're using backend with other commands - don't exit
            print(f"Running with backend: {BackendManager.get_backend().name}")
            return False
        else:
            # Just setting the backend - exit after this
            return True
    
    else:
        # Try to process as a compound command
        parts = backend_arg.split()
        if len(parts) >= 1:
            try:
                BackendManager.set_backend(parts[0])
                print(f"Backend set to {BackendManager.get_backend().name}")
                
                # If there are more parts, try to handle them as commands
                if len(parts) > 1:
                    if parts[1].lower() == 'info':
                        backend = BackendManager.get_backend()
                        print(f"Supports GPU: {backend.supports_gpu}")
                        print(f"Available math functions: {', '.join(backend.get_math_functions().keys())}")
                
                # Check if there are other actions to perform
                has_other_action = (
                    hasattr(args, 'input_file') and args.input_file or
                    hasattr(args, 'keyframes') and args.keyframes or
                    hasattr(args, 'visualize') and args.visualize or
                    hasattr(args, 'scene') and args.scene
                )
                
                if has_other_action:
                    # We're using backend with other commands - don't exit
                    print(f"Running with backend: {BackendManager.get_backend().name}")
                    return False
                else:
                    # Just setting the backend - exit after this
                    return True
            except Exception as e:
                print(f"Error processing backend command: {e}")
                return True
    
    return True


def generate_scene_cmd(args: argparse.Namespace) -> None:
    """Handle the generate-scene command.
    
    Args:
        args: Command-line arguments with scene generation options
    """
    if not args.generate_scene:
        print("Error: No output file specified for generate-scene command.")
        return
    
    output_file = args.generate_scene
    
    # Create a solver
    solver = Solver(name="GeneratedExample")
    solver.set_metadata("description", "Generated example scene")
    solver.set_metadata("author", "SplinalTap")
    solver.set_variable("pi", 3.14159)
    
    # If we have an input file, try to load it
    if hasattr(args, 'input_file') and args.input_file:
        try:
            # Load the input file
            source_solver = Solver.load(args.input_file)
            
            # Copy metadata and variables
            solver.name = source_solver.name
            solver.metadata = source_solver.metadata.copy()
            solver.variables = source_solver.variables.copy()
            
            # Copy splines
            for name, spline in source_solver.splines.items():
                new_spline = solver.create_spline(name)
                
                # Copy channels
                for channel_name, channel in spline.channels.items():
                    new_channel = new_spline.add_channel(
                        name=channel_name,
                        interpolation=channel.interpolation,
                        min_max=channel.min_max
                    )
                    
                    # Copy keyframes
                    for kf in channel.keyframes:
                        new_channel.add_keyframe(
                            at=kf.at,
                            value=kf.value(kf.at, {}),  # Evaluate to get value
                            interpolation=kf.interpolation,
                            control_points=kf.control_points,
                            derivative=kf.derivative
                        )
        except Exception as e:
            print(f"Warning: Could not load input file: {e}")
            # Continue with generating a default scene
    
    # If we have keyframes, use them to create a new spline
    if hasattr(args, 'keyframes') and args.keyframes:
        # Parse the keyframes
        try:
            keyframes = []
            for kf_str in args.keyframes:
                # First, split on @ to separate the position:value part from the method part
                if '@' in kf_str and ':' in kf_str.split('@')[0]:
                    main_part, method_part = kf_str.split('@', 1)
                    method_name, params = parse_method_parameters(method_part)
                else:
                    main_part = kf_str
                    method_name, params = None, None
                
                # Now parse the position:value part
                parts = main_part.split(':')
                if len(parts) < 2:
                    raise ValueError(f"Invalid keyframe format: {kf_str}")
                
                position = float(parts[0])
                value = parts[1]
                
                # Handle method-specific parameters
                derivative = None
                control_points = None
                
                if method_name == "hermite" and params and "deriv" in params:
                    derivative = float(params["deriv"])
                elif method_name == "bezier" and params:
                    # Parse control points
                    if "cp" in params:
                        cp_str = params["cp"].strip()
                        cp_values = [float(p.strip()) for p in cp_str.split(',')]
                        if len(cp_values) == 4:
                            control_points = cp_values
                
                # Sanitize the value if it's a string expression
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except ValueError:
                        value = sanitize_for_ast(value)
                
                keyframes.append((position, value, derivative, control_points, method_name))
            
            # Create a new spline with the keyframes
            if not solver.splines:
                spline = solver.create_spline("position")
                channel = spline.add_channel("x")
                
                # Add the keyframes
                for pos, val, deriv, cp, method in keyframes:
                    channel.add_keyframe(
                        at=pos,
                        value=val,
                        interpolation=method,
                        control_points=cp,
                        derivative=deriv
                    )
            else:
                # Add keyframes to existing spline
                for name, spline in solver.splines.items():
                    if "position" in name.lower():
                        # Find an x channel or similar
                        for ch_name, channel in spline.channels.items():
                            if ch_name.lower() in ('x', 'value', 'default'):
                                # Add the keyframes
                                for pos, val, deriv, cp, method in keyframes:
                                    channel.add_keyframe(
                                        at=pos,
                                        value=val,
                                        interpolation=method,
                                        control_points=cp,
                                        derivative=deriv
                                    )
                                break
                        break
        except Exception as e:
            print(f"Warning: Could not parse keyframes: {e}")
    
    # If we still don't have any splines, create a default one
    if not solver.splines:
        # Create a default position spline
        position = solver.create_spline("position")
        
        # Add x, y, z channels
        dimensions = 3
        if hasattr(args, 'dimensions'):
            try:
                dimensions = int(args.dimensions)
            except ValueError:
                pass
        
        channel_names = ['x', 'y', 'z'][:dimensions]
        
        for i, name in enumerate(channel_names):
            channel = position.add_channel(name)
            
            # Add some default keyframes
            channel.add_keyframe(at=0.0, value=0)
            channel.add_keyframe(at=0.5, value=f"sin(@ * pi * {i+1})" if i > 0 else 5)
            channel.add_keyframe(at=1.0, value=0)
    
    # Determine output format
    content_type = getattr(args, 'content_type', 'json')
    
    # Save the scene
    if content_type == 'yaml':
        # Make sure we have the yaml extension
        if not output_file.endswith(('.yml', '.yaml')):
            output_file += '.yaml'
        solver.save(output_file, format='yaml')
    else:
        # Default to JSON
        if not output_file.endswith('.json'):
            output_file += '.json'
        solver.save(output_file, format='json')
    
    print(f"Generated scene saved to {output_file}")


def create_parser():
    """Create and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="SplinalTap - Keyframe interpolation tool")
    
    # Input/output options
    parser.add_argument('--input-file', type=str, help="Input file in JSON format")
    parser.add_argument('--output-file', type=str, help="Output file path")
    parser.add_argument('--content-type', type=str, choices=['json', 'csv', 'yaml', 'text'], default='json',
                        help="Output content type (default: json)")
    
    # Keyframe and sampling options
    parser.add_argument('--keyframes', type=str, nargs='+', 
                        help="Keyframes in format '@:value[@method{parameters}]'")
    parser.add_argument('--samples', type=str, nargs='+',
                        help="Sample points or count (integer for count, float for points)")
    parser.add_argument('--range', type=str, help="Sample range min,max (for integer sample count)")
    parser.add_argument('--use-indices', action='store_true', help="Use absolute indices instead of normalized 0-1 range")
    parser.add_argument('--variables', type=str, help="Variables in format 'name1=value1,name2=value2,...'")
    
    return parser

def main():
    """Main entry point for the command-line interface."""
    # Create argument parser
    parser = create_parser()
    
    # Commands
    parser.add_argument('--visualize', action='store_true', help="Visualize the interpolation")
    parser.add_argument('--scene', type=str, help="Scene command (info/ls/convert/extract) with args")
    parser.add_argument('--backend', type=str, help="Backend to use or backend command")
    parser.add_argument('--generate-scene', type=str, help="Generate a scene file at the specified path")
    parser.add_argument('--dimensions', type=int, help="Number of dimensions for generated scene (default: 3)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle backend commands first
    if hasattr(args, 'backend') and args.backend:
        if backend_cmd(args):
            return
    
    # Handle scene commands
    if hasattr(args, 'scene') and args.scene:
        scene_cmd(args)
        return
    
    # Handle generate-scene command
    if hasattr(args, 'generate_scene') and args.generate_scene:
        generate_scene_cmd(args)
        return
    
    # Require either input file or keyframes
    if not args.input_file and not args.keyframes:
        parser.error("Either --input-file or --keyframes is required")
    
    # Create solver from arguments
    solver = create_solver_from_args(args)
    
    # Handle visualize command
    if args.visualize:
        visualize_solver(solver, args)
        return
    
    # Default behavior: sample at specified points
    results = sample_solver(solver, args)
    
    # Output the results
    if args.output_file:
        save_results(results, args)
    else:
        print_results(results)


if __name__ == "__main__":
    main()