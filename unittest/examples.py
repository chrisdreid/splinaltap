"""
Example usage of the splinaltap library.
"""

from .solver import KeyframeSolver
from .spline import Spline
from .channel import Channel, Keyframe
from .expression import ExpressionEvaluator
from .backends import BackendManager
import matplotlib.pyplot as plt
import time
import os

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

def basic_interpolation_example():
    """Create a basic interpolation example with visualization."""
    # Create a solver with a spline and channel
    solver = KeyframeSolver(name="Basic Example")
    spline = solver.create_spline("main")
    channel = spline.add_channel("value", interpolation="cubic")
    
    # Add keyframes with expressions
    channel.add_keyframe(0.0, 0)
    channel.add_keyframe(0.25, "sin(@) + 1")  # '@' is the current position
    channel.add_keyframe(0.5, "3 + cos(@ * pi)")  # Built-in constants like pi are available
    channel.add_keyframe(0.75, "5 * @ - 2")
    channel.add_keyframe(1.0, 10)
    
    # Visualize the interpolation using different methods
    values = {}
    samples = [i/100 for i in range(101)]
    
    for method in ["linear", "cubic", "bezier", "ease_in_out"]:
        temp_channel = Channel(interpolation=method)
        for kf in channel.keyframes:
            temp_channel.add_keyframe(kf.at, kf.value, control_points=kf.control_points, derivative=kf.derivative)
        
        values[method] = [temp_channel.get_value(pos) for pos in samples]
    
    # Visualize using matplotlib directly
    plt.figure(figsize=(10, 6))
    for method, vals in values.items():
        plt.plot(samples, vals, label=method)
    
    plt.title("Interpolation Methods Comparison")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("interpolation_comparison.png")
    
    return solver

def external_channels_example():
    """Example showing how to use external channels."""
    # Create a solver with a spline and channel
    solver = KeyframeSolver(name="External Channels Example")
    spline = solver.create_spline("main")
    
    # Create a channel with expressions referencing external channels
    channel = spline.add_channel("value", interpolation="cubic")
    channel.add_keyframe(0.0, 0)
    channel.add_keyframe(0.3, "a * sin(@) + b")
    channel.add_keyframe(0.7, "a * cos(@) + c")
    channel.add_keyframe(1.0, 10)
    
    # Evaluate with different external channel values
    samples = [i/100 for i in range(101)]
    
    # Two different sets of external channels
    ext_channels_1 = {"a": 1.0, "b": 0.5, "c": 1.0}
    ext_channels_2 = {"a": 2.0, "b": 0.0, "c": 3.0}
    
    # Evaluate with each set
    values_1 = [channel.get_value(pos, ext_channels_1) for pos in samples]
    values_2 = [channel.get_value(pos, ext_channels_2) for pos in samples]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(samples, values_1, label="External Channels Set 1")
    plt.plot(samples, values_2, label="External Channels Set 2")
    
    # Plot keyframes
    keyframe_pos = [kf.at for kf in channel.keyframes]
    keyframe_vals = [channel.get_value(pos, ext_channels_1) for pos in keyframe_pos]
    plt.scatter(keyframe_pos, keyframe_vals, color='black', s=100,
                facecolors='none', edgecolors='black', label='Keyframes (Set 1)')
    
    plt.title("External Channels Comparison")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return solver

def bezier_control_points_example():
    """Example showing how to use Bezier control points."""
    # Create a solver with a spline and channel
    solver = KeyframeSolver(name="Bezier Example")
    spline = solver.create_spline("main")
    channel = spline.add_channel("value", interpolation="bezier")
    
    # Add keyframes with bezier control points
    channel.add_keyframe(0.0, 0)
    channel.add_keyframe(0.4, 5.0, {"cp": [0.42, 6.0, 0.48, 7.0]})
    channel.add_keyframe(0.7, 2.0, {"cp": [0.72, 1.0, 0.78, 0.5]})
    channel.add_keyframe(1.0, 10)
    
    # Evaluate and plot
    samples = [i/100 for i in range(101)]
    values = [channel.get_value(pos) for pos in samples]
    
    plt.figure(figsize=(10, 6))
    plt.plot(samples, values)
    
    # Plot keyframes
    keyframe_pos = [kf.at for kf in channel.keyframes]
    keyframe_vals = [channel.get_value(kf.at) for kf in channel.keyframes]
    plt.scatter(keyframe_pos, keyframe_vals, color='red', s=100)
    
    plt.title("Bezier Interpolation with Control Points")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    
    return solver

def multidimensional_example():
    """Example showing how to use multiple channels in a spline."""
    # Create a solver with a position spline (x, y, z channels)
    solver = KeyframeSolver(name="Multidimensional Example")
    position = solver.create_spline("position")
    
    # Add x, y, z channels
    x_channel = position.add_channel("x", interpolation="cubic")
    y_channel = position.add_channel("y", interpolation="cubic")
    z_channel = position.add_channel("z", interpolation="linear")
    
    # Add keyframes to each channel
    x_channel.add_keyframe(0.0, 0.0)
    x_channel.add_keyframe(0.5, 10.0)
    x_channel.add_keyframe(1.0, 0.0)
    
    y_channel.add_keyframe(0.0, 0.0)
    y_channel.add_keyframe(0.3, "sin(@) * 10")
    y_channel.add_keyframe(0.7, "cos(@) * 10")
    y_channel.add_keyframe(1.0, 0.0)
    
    z_channel.add_keyframe(0.0, 0.0)
    z_channel.add_keyframe(0.25, 5.0)
    z_channel.add_keyframe(0.75, 5.0)
    z_channel.add_keyframe(1.0, 0.0)
    
    # Sample all channels
    samples = [i/100 for i in range(101)]
    x_values = [x_channel.get_value(pos) for pos in samples]
    y_values = [y_channel.get_value(pos) for pos in samples]
    z_values = [z_channel.get_value(pos) for pos in samples]
    
    # 3D plot
    ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
    ax.plot(x_values, y_values, z_values, label="3D Path")
    
    # Plot keyframe points for clarity
    keyframe_pos = sorted(set([kf.at for kf in x_channel.keyframes] + 
                           [kf.at for kf in y_channel.keyframes] + 
                           [kf.at for kf in z_channel.keyframes]))
    
    keyframe_x = [x_channel.get_value(pos) for pos in keyframe_pos]
    keyframe_y = [y_channel.get_value(pos) for pos in keyframe_pos]
    keyframe_z = [z_channel.get_value(pos) for pos in keyframe_pos]
    
    ax.scatter(keyframe_x, keyframe_y, keyframe_z, color='red', s=100, label="Keyframes")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Position Path")
    plt.tight_layout()
    plt.show()
    
    return solver

def solver_serialization_example():
    """Example showing how to serialize and deserialize a solver."""
    # Create a solver with multiple splines
    solver = KeyframeSolver(name="Animation")
    solver.set_metadata("description", "An example animation")
    solver.set_metadata("author", "SplinalTap")
    
    # Create a position spline
    position = solver.create_spline("position")
    position.add_channel("x").add_keyframe(0.0, 0.0).add_keyframe(1.0, 10.0)
    position.add_channel("y").add_keyframe(0.0, 0.0).add_keyframe(1.0, 5.0)
    position.add_channel("z").add_keyframe(0.0, 0.0).add_keyframe(1.0, 0.0)
    
    # Create a rotation spline
    rotation = solver.create_spline("rotation")
    rotation.add_channel("x").add_keyframe(0.0, 0.0).add_keyframe(1.0, 360.0)
    rotation.add_channel("y").add_keyframe(0.0, 0.0).add_keyframe(1.0, 0.0)
    rotation.add_channel("z").add_keyframe(0.0, 0.0).add_keyframe(1.0, 0.0)
    
    # Create a scale spline
    scale = solver.create_spline("scale")
    scale.add_channel("uniform").add_keyframe(0.0, 1.0).add_keyframe(0.5, 2.0).add_keyframe(1.0, 1.0)
    
    # Save in different formats
    temp_dir = "/tmp"
    formats = [("json", "animation.json")]
    
    if HAS_YAML:
        formats.append(("yaml", "animation.yaml"))
    
    saved_files = []
    for format_name, filename in formats:
        filepath = os.path.join(temp_dir, filename)
        try:
            solver.save(filepath, format=format_name)
            saved_files.append(filepath)
            print(f"Saved solver in {format_name} format to {filepath}")
        except Exception as e:
            print(f"Error saving in {format_name} format: {e}")
    
    # Load back the JSON version
    if saved_files:
        json_file = os.path.join(temp_dir, "animation.json")
        loaded_solver = KeyframeSolver.from_file(json_file)
        print(f"Loaded solver: {loaded_solver.name}")
        print(f"Metadata: {loaded_solver.metadata}")
        print(f"Spline names: {loaded_solver.get_spline_names()}")
        
        # Sample all splines at a specific position
        pos = 0.5
        for spline_name in loaded_solver.get_spline_names():
            spline = loaded_solver.get_spline(spline_name)
            values = {name: spline.get_channel(name).get_value(pos) for name in spline.get_channel_names()}
            print(f"{spline_name} at pos={pos}: {values}")
    
    return solver

def backends_example():
    """Example demonstrating different compute backends."""
    # Create a solver with a spline and channel
    solver = KeyframeSolver(name="Backend Example")
    spline = solver.create_spline("main")
    channel = spline.add_channel("value", interpolation="cubic")
    
    # Add keyframes with complex expressions
    channel.add_keyframe(0.0, 0.0)
    channel.add_keyframe(0.25, "sin(@ * 10)")
    channel.add_keyframe(0.5, "sin(@ * 5) * cos(@ * 10)")
    channel.add_keyframe(1.0, 10.0)
    
    # Sample with different backends and measure performance
    num_samples = 100000  # Large number to compare performance
    samples = [i/num_samples for i in range(num_samples)]
    
    backends = []
    backends.append(("python", "Pure Python"))
    if HAS_NUMPY:
        backends.append(("numpy", "NumPy (CPU)"))
    
    try:
        import cupy
        backends.append(("cupy", "CuPy (GPU)"))
    except ImportError:
        pass
    
    for backend_name, label in backends:
        # Set backend
        BackendManager.set_backend(backend_name)
        print(f"Using {label} backend:")
        
        # Time the sampling
        start_time = time.time()
        result = [channel.get_value(pos) for pos in samples]
        end_time = time.time()
        
        # Report performance
        elapsed = end_time - start_time
        print(f"  Sampled {num_samples} points in {elapsed:.4f} seconds")
        print(f"  {num_samples/elapsed:.0f} samples per second")
        
        # Print some sample values
        sample_indices = [0, 1000, 10000, 50000, 99999]
        print("  Sample values:", end=" ")
        for idx in sample_indices:
            if idx < len(result):
                print(f"{float(result[idx]):.2f}", end=" ")
        print()
    
    # Reset to best backend
    BackendManager.use_best_available()
    return solver

if __name__ == "__main__":
    # Run all examples
    basic_interpolation_example()
    external_channels_example()
    bezier_control_points_example()
    multidimensional_example()
    solver_serialization_example()
    
    # Only run backend examples if numpy is available
    if HAS_NUMPY:
        backends_example()
    else:
        print("Skipping backend examples (numpy not available)")