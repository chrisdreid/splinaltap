"""
Demo script to show the new plotting functionality in SplinalTap.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from splinaltap import KeyframeSolver

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib is not installed. Install it with: pip install matplotlib")
    print("Continuing with reduced functionality...")

def spline_plot_demo():
    """Demonstrate plotting a single spline."""
    if not HAS_MATPLOTLIB:
        print("Cannot run plot demo without matplotlib.")
        return

    print("\n=== Spline Plot Demo ===")
    
    # Create a solver and spline
    solver = KeyframeSolver(name="SplinePlotDemo")
    spline = solver.create_spline("wave")
    
    # Create multiple channels
    sine = spline.add_channel("sine")
    cosine = spline.add_channel("cosine")
    
    # Add keyframes
    sine.add_keyframe(at=0.0, value=0.0)
    sine.add_keyframe(at=0.25, value=1.0)
    sine.add_keyframe(at=0.5, value=0.0)
    sine.add_keyframe(at=0.75, value=-1.0)
    sine.add_keyframe(at=1.0, value=0.0)
    
    cosine.add_keyframe(at=0.0, value=1.0)
    cosine.add_keyframe(at=0.25, value=0.0)
    cosine.add_keyframe(at=0.5, value=-1.0)
    cosine.add_keyframe(at=0.75, value=0.0)
    cosine.add_keyframe(at=1.0, value=1.0)
    
    # Plot the spline
    print("Plotting the 'wave' spline with sine and cosine channels...")
    spline.plot(samples=200, title="Sine and Cosine Waves")
    
    # Plot with filtered channels and dark theme
    print("Plotting only the sine channel with dark theme...")
    spline.plot(samples=200, filter_channels=["sine"], theme="dark", title="Sine Wave (Dark Theme)")
    
    print("Spline plotting complete.")

def solver_plot_demo():
    """Demonstrate plotting a solver with multiple splines."""
    if not HAS_MATPLOTLIB:
        print("Cannot run plot demo without matplotlib.")
        return

    print("\n=== Solver Plot Demo ===")
    
    # Create a solver
    solver = KeyframeSolver(name="AnimationCurves")
    
    # Create position spline with x, y, z channels
    position = solver.create_spline("position")
    x = position.add_channel("x")
    y = position.add_channel("y")
    z = position.add_channel("z")
    
    # Add keyframes for position channels
    x.add_keyframe(at=0.0, value=0.0)
    x.add_keyframe(at=0.5, value=5.0)
    x.add_keyframe(at=1.0, value=0.0)
    
    y.add_keyframe(at=0.0, value=0.0)
    y.add_keyframe(at=0.25, value=2.5)
    y.add_keyframe(at=0.75, value=-2.5)
    y.add_keyframe(at=1.0, value=0.0)
    
    z.add_keyframe(at=0.0, value=0.0)
    z.add_keyframe(at=1.0, value=10.0)
    
    # Create rotation spline with angle channel
    rotation = solver.create_spline("rotation")
    angle = rotation.add_channel("angle")
    
    # Add keyframes for rotation channel
    angle.add_keyframe(at=0.0, value=0.0)
    angle.add_keyframe(at=1.0, value=360.0)
    
    # Create a scale spline with factor channel
    scale = solver.create_spline("scale")
    factor = scale.add_channel("factor")
    
    # Add keyframes for scale channel
    factor.add_keyframe(at=0.0, value=1.0)
    factor.add_keyframe(at=0.5, value=2.0)
    factor.add_keyframe(at=1.0, value=1.0)
    
    # Plot the entire solver
    print("Plotting all splines and channels in the solver...")
    solver.plot(samples=100)
    
    # Plot with filtered splines and channels
    print("Plotting only position.x and rotation.angle with dark theme...")
    solver.plot(
        samples=100,
        filter_channels={
            "position": ["x"],
            "rotation": ["angle"]
        },
        theme="dark"
    )
    
    print("Solver plotting complete.")

def expression_plot_demo():
    """Demonstrate plotting with expressions and cross-references."""
    if not HAS_MATPLOTLIB:
        print("Cannot run plot demo without matplotlib.")
        return

    print("\n=== Expression Plot Demo ===")
    
    # Create a solver
    solver = KeyframeSolver(name="ExpressionDemo")
    
    # Create splines
    base = solver.create_spline("base")
    derived = solver.create_spline("derived")
    
    # Add channels
    value = base.add_channel("value")
    amplified = derived.add_channel("amplified")
    squared = derived.add_channel("squared")
    
    # Add keyframes to base channel
    value.add_keyframe(at=0.0, value=0.0)
    value.add_keyframe(at=0.5, value=5.0)
    value.add_keyframe(at=1.0, value=0.0)
    
    # Set up publishing
    solver.set_publish("base.value", ["derived.amplified", "derived.squared"])
    
    # Add keyframes with expressions referencing the base channel
    amplified.add_keyframe(at=0.0, value="base.value * 2")
    amplified.add_keyframe(at=1.0, value="base.value * 2")
    
    squared.add_keyframe(at=0.0, value="base.value * base.value")
    squared.add_keyframe(at=1.0, value="base.value * base.value")
    
    # Plot the solver
    print("Plotting solver with expression-based channels...")
    solver.plot(samples=100)
    
    print("Expression plotting complete.")

def main():
    """Run all the plot demos."""
    print("SplinalTap Plotting Functionality Demo")
    
    # Run the demo functions
    spline_plot_demo()
    solver_plot_demo()
    expression_plot_demo()
    
    print("\nAll plot demos completed.")

if __name__ == "__main__":
    main()