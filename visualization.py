"""
Visualization tools for the splinaltap library.
"""

from .backends import BackendManager

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

# Check if NumPy is available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def plot_interpolation_comparison(interpolator, t_values, methods=None, channels=None, title="Interpolation Methods Comparison"):
    """
    Plot a comparison of different interpolation methods.
    
    Args:
        interpolator: The KeyframeInterpolator instance to use
        t_values: List of time values to evaluate
        methods: List of interpolation methods to compare (defaults to all available methods)
        channels: Dictionary of channel values to use in expressions
        title: Title for the plot
    
    Returns:
        The matplotlib figure or None if matplotlib is not available
    """
    if not has_matplotlib:
        raise ImportError("Visualization requires matplotlib")
    
    channels = channels or {}
    if methods is None:
        methods = ["nearest", "linear", "polynomial", "quadratic", 
                   "hermite", "bezier", "pchip", "cubic"]
        try:
            # Test if numpy is available for gaussian method
            from . import methods
            if methods.np is not None:
                methods.append("gaussian")
        except (ImportError, AttributeError):
            pass
    
    # Get keyframe points for scatter plot
    keyframe_points = interpolator._get_keyframe_points(channels)
    keyframe_t = [p[0] for p in keyframe_points]
    keyframe_values = [p[1] for p in keyframe_points]

    # Convert to standard Python lists first to avoid backend issues
    t_values_list = [float(t) for t in t_values]
    
    # Convert lists to numpy for matplotlib
    if HAS_NUMPY:
        t_values_np = np.array(t_values_list)
    else:
        t_values_np = t_values_list

    fig = plt.figure(figsize=(12, 8))
    for method in methods:
        try:
            # Sample values using current backend
            values = interpolator.sample_range(min(t_values_list), max(t_values_list), len(t_values_list), method, channels)
            
            # Convert to standard Python list first
            if hasattr(values, 'tolist'):
                values_list = values.tolist()
            else:
                values_list = [float(v) for v in values]
                
            # Convert to numpy for matplotlib
            if HAS_NUMPY:
                values_np = np.array(values_list)
            else:
                values_np = values_list
            
            plt.plot(t_values_np, values_np, label=method.capitalize())
        except Exception as e:
            print(f"Skipping {method} due to: {e}")

    # Add circles at keyframe points
    plt.scatter(keyframe_t, keyframe_values, color='black', s=100, 
                facecolors='none', edgecolors='black', label='Keyframes')

    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
    return fig

def plot_single_interpolation(interpolator, t_values, method="cubic", channels=None, title=None):
    """
    Plot a single interpolation method.
    
    Args:
        interpolator: The KeyframeInterpolator instance to use
        t_values: List of time values to evaluate
        method: Interpolation method to use
        channels: Dictionary of channel values to use in expressions
        title: Title for the plot (defaults to method name)
    
    Returns:
        The matplotlib figure or None if matplotlib is not available
    """
    if not has_matplotlib:
        raise ImportError("Visualization requires matplotlib")
    
    channels = channels or {}
    title = title or f"{method.capitalize()} Interpolation"
    
    # Get keyframe points for scatter plot
    keyframe_points = interpolator._get_keyframe_points(channels)
    keyframe_t = [p[0] for p in keyframe_points]
    keyframe_values = [p[1] for p in keyframe_points]
    
    # Convert to standard Python lists first to avoid backend issues
    t_values_list = [float(t) for t in t_values]
    
    # Convert lists to numpy for matplotlib
    if HAS_NUMPY:
        t_values_np = np.array(t_values_list)
    else:
        t_values_np = t_values_list
    
    # Sample values using current backend
    values = interpolator.sample_range(min(t_values_list), max(t_values_list), len(t_values_list), method, channels)
    
    # Convert to standard Python list first
    if hasattr(values, 'tolist'):
        values_list = values.tolist()
    else:
        values_list = [float(v) for v in values]
        
    # Convert to numpy for matplotlib
    if HAS_NUMPY:
        values_np = np.array(values_list)
    else:
        values_np = values_list
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(t_values_np, values_np, label=method.capitalize())
    plt.scatter(keyframe_t, keyframe_values, color='black', s=100, 
                facecolors='none', edgecolors='black', label='Keyframes')
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
    return fig