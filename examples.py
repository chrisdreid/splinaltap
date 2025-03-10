"""
Example usage of the splinaltap library.
"""

from .interpolator import KeyframeInterpolator
from .visualization import plot_interpolation_comparison, plot_single_interpolation
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # Run all examples
    basic_interpolation_example()
    channels_example()
    bezier_control_points_example()
    time_based_example()