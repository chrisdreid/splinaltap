"""
Example usage of the splinaltap library.
"""

from .interpolator import KeyframeInterpolator
from .visualization import plot_interpolation_comparison, plot_single_interpolation
import matplotlib.pyplot as plt

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
    # Create a KeyframeInterpolator instance with 10 indices
    interpolator = KeyframeInterpolator(10)
    
    # Add keyframes with expressions
    interpolator.set_keyframe(0.0, 0)
    interpolator.set_keyframe(2.5, "sin(t) + 1")  # 't' is the current time
    interpolator.set_keyframe(5.7, "pow(t, 2)")
    interpolator.set_keyframe(10.0, 10)
    
    # Define a variable
    interpolator.set_variable("amplitude", 2.5)
    interpolator.set_keyframe(7.0, "amplitude * sin(t)")
    
    # Evaluate at various points
    t_values = [i * 0.1 for i in range(101)]
    
    # Plot a single interpolation method
    fig = plot_single_interpolation(interpolator, t_values, "cubic")
    plt.show()
    
    # Compare different interpolation methods
    fig = plot_interpolation_comparison(interpolator, t_values)
    plt.show()
    
    return interpolator

def channels_example():
    """Example showing how to use channels in expressions."""
    interpolator = KeyframeInterpolator(10)
    
    # Define keyframes that use channel values
    interpolator.set_keyframe(0.0, 0)
    interpolator.set_keyframe(3.0, "a * sin(t) + b")
    interpolator.set_keyframe(7.0, "a * cos(t) + c")
    interpolator.set_keyframe(10.0, 10)
    
    # Evaluate with different channel values
    t_values = [i * 0.1 for i in range(101)]
    channels_1 = {"a": 1.0, "b": 0.5, "c": 1.0}
    channels_2 = {"a": 2.0, "b": 0.0, "c": 3.0}
    
    # Plot with different channel values
    plt.figure(figsize=(12, 6))
    
    values_1 = [interpolator.get_value(t, "cubic", channels_1) for t in t_values]
    plt.plot(t_values, values_1, label="Channels Set 1")
    
    values_2 = [interpolator.get_value(t, "cubic", channels_2) for t in t_values]
    plt.plot(t_values, values_2, label="Channels Set 2")
    
    # Get keyframe points for the first channel set
    keyframe_points = interpolator._get_keyframe_points(channels_1)
    keyframe_t = [p[0] for p in keyframe_points]
    keyframe_values = [p[1] for p in keyframe_points]
    
    # Add circles at keyframe points
    plt.scatter(keyframe_t, keyframe_values, color='black', s=100, 
                facecolors='none', edgecolors='black', label='Keyframes (Set 1)')
    
    plt.legend()
    plt.title("Channel Values Comparison")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    
    return interpolator

def bezier_control_points_example():
    """Example showing how to use Bezier control points."""
    interpolator = KeyframeInterpolator(10)
    
    # Set keyframes with control points for Bezier interpolation
    interpolator.set_keyframe(0.0, 0)
    interpolator.set_keyframe(4.0, 5.0, derivative=None, control_points=(4.2, 6.0, 4.8, 7.0))
    interpolator.set_keyframe(7.0, 2.0, derivative=None, control_points=(7.2, 1.0, 7.8, 0.5))
    interpolator.set_keyframe(10.0, 10)
    
    # Evaluate at various points
    t_values = [i * 0.1 for i in range(101)]
    
    # Plot with bezier interpolation
    fig = plot_single_interpolation(interpolator, t_values, "bezier", 
                                   title="Bezier Interpolation with Control Points")
    plt.show()
    
    return interpolator

def time_based_example():
    """Example using time in milliseconds instead of normalized indices."""
    # Create a time-based interpolator (no fixed indices)
    interpolator = KeyframeInterpolator()
    
    # Add keyframes at specific millisecond times
    interpolator.set_keyframe(1000.0, 0)
    interpolator.set_keyframe(1234.567, 10)
    interpolator.set_keyframe(2345.678, "pi * sin(t/1000)")
    interpolator.set_keyframe(3500.0, 15)
    
    # Get time range
    t_min, t_max = interpolator.get_time_range()
    print(f"Time range: {t_min} ms to {t_max} ms")
    
    # Sample into an array (100 samples)
    values = interpolator.sample_range(t_min, t_max, 100, "cubic")
    
    # Create time values for plotting
    t_values = [t_min + i * (t_max - t_min) / 99 for i in range(100)]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, values)
    plt.title("Time-Based Interpolation (milliseconds)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.grid(True)
    
    # Highlight keyframe points
    keyframe_points = interpolator._get_keyframe_points()
    keyframe_t = [p[0] for p in keyframe_points]
    keyframe_values = [p[1] for p in keyframe_points]
    plt.scatter(keyframe_t, keyframe_values, color='red', s=100)
    
    plt.show()
    
    return interpolator, values

def scene_example():
    """Example showing how to work with scenes that contain multiple interpolators."""
    from .scene import Scene
    import os
    
    # Create a scene
    scene = Scene(name="AnimationScene")
    scene.set_metadata("description", "An example animation scene")
    scene.set_metadata("author", "Splinaltap")
    
    # Create and add multiple interpolators
    
    # Position X interpolator
    position_x = KeyframeInterpolator()
    position_x.set_keyframe(0.0, 0.0)
    position_x.set_keyframe(1000.0, 100.0)
    position_x.set_keyframe(2000.0, 50.0)
    position_x.set_keyframe(3000.0, 75.0)
    scene.add_interpolator("position_x", position_x)
    
    # Position Y interpolator
    position_y = KeyframeInterpolator()
    position_y.set_keyframe(0.0, 0.0)
    position_y.set_keyframe(1000.0, "sin(t/100) * 50")
    position_y.set_keyframe(2000.0, 100.0)
    position_y.set_keyframe(3000.0, 25.0)
    scene.add_interpolator("position_y", position_y)
    
    # Scale interpolator
    scale = KeyframeInterpolator()
    scale.set_keyframe(0.0, 1.0)
    scale.set_keyframe(1500.0, 2.0)
    scale.set_keyframe(3000.0, 0.5)
    scene.add_interpolator("scale", scale)
    
    # Rotation interpolator with bezier control
    rotation = KeyframeInterpolator()
    rotation.set_keyframe(0.0, 0.0)
    rotation.set_keyframe(1500.0, 180.0, control_points=(500.0, 20.0, 1000.0, 160.0))
    rotation.set_keyframe(3000.0, 360.0)
    scene.add_interpolator("rotation", rotation)
    
    # Save scene in different formats
    temp_dir = "/tmp"
    formats = [
        ("json", "scene.json"), 
        ("python", "scene.py"),
        ("pickle", "scene.pkl")
    ]
    
    # Add optional formats if available
    if HAS_NUMPY:
        formats.append(("numpy", "scene.npz"))
    if HAS_YAML:
        formats.append(("yaml", "scene.yaml"))
    
    saved_files = []
    for format_name, filename in formats:
        filepath = os.path.join(temp_dir, filename)
        try:
            scene.save(filepath, format=format_name)
            saved_files.append(filepath)
            print(f"Saved scene in {format_name} format to {filepath}")
        except Exception as e:
            print(f"Error saving in {format_name} format: {e}")
    
    # Load back the JSON version
    if saved_files:
        json_file = os.path.join(temp_dir, "scene.json")
        loaded_scene = Scene.load(json_file)
        print(f"Loaded scene: {loaded_scene.name}")
        print(f"Metadata: {loaded_scene.metadata}")
        print(f"Interpolator names: {loaded_scene.get_interpolator_names()}")
        
        # Sample all interpolators at a specific time
        t = 1500.0
        results = {}
        for name in loaded_scene.get_interpolator_names():
            interp = loaded_scene.get_interpolator(name)
            results[name] = interp.get_value(t, "cubic")
        
        print(f"Values at t={t}ms: {results}")
    
    return scene

def backends_example():
    """Example demonstrating different compute backends."""
    from .backends import BackendManager, PythonBackend, NumpyBackend, CupyBackend
    import time
    
    # Create a large interpolation
    interpolator = KeyframeInterpolator()
    interpolator.set_keyframe(0.0, 0.0)
    interpolator.set_keyframe(100.0, "sin(t/10)")
    interpolator.set_keyframe(200.0, "sin(t/5) * cos(t/10)")
    interpolator.set_keyframe(1000.0, 10.0)
    
    # Sample with different backends and measure performance
    num_samples = 100000  # Large number to compare performance
    
    backends = []
    if PythonBackend.is_available:
        backends.append(("python", "Pure Python"))
    if NumpyBackend.is_available:
        backends.append(("numpy", "NumPy (CPU)"))
    if CupyBackend.is_available:
        backends.append(("cupy", "CuPy (GPU)"))
    
    for backend_name, label in backends:
        # Set backend
        BackendManager.set_backend(backend_name)
        print(f"Using {label} backend:")
        
        # Time the sampling
        start_time = time.time()
        result = interpolator.sample_range(0.0, 1000.0, num_samples, method="linear")
        end_time = time.time()
        
        # Report performance
        elapsed = end_time - start_time
        print(f"  Sampled {num_samples} points in {elapsed:.4f} seconds")
        print(f"  {num_samples/elapsed:.0f} samples per second")
        
        # Print some sample values
        samples = [0, 1000, 10000, 50000, 99999]
        print("  Sample values:", end=" ")
        try:
            for idx in samples:
                if idx < len(result):
                    print(f"{float(result[idx]):.2f}", end=" ")
        except Exception as e:
            print(f"Error accessing results: {e}")
        print()
    
    # Reset to best backend
    BackendManager.use_best_available()
    return interpolator

if __name__ == "__main__":
    # Run all examples
    basic_interpolation_example()
    channels_example()
    bezier_control_points_example()
    time_based_example()
    scene_example()
    
    # Only run backend examples if numpy is available
    try:
        import numpy
        backends_example()
    except ImportError:
        print("Skipping backend examples (numpy not available)")