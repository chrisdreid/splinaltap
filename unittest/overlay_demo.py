#!/usr/bin/env python3
"""
Demo script for SplinalTap plot overlay functionality.
Compares overlay=True vs overlay=False options.
"""

from splinaltap.solver import KeyframeSolver
from splinaltap.spline import Spline

def main():
    # Create a complex solver for demonstration
    solver = KeyframeSolver(name="OverlayDemo")
    
    # Create two splines with multiple channels
    position = solver.create_spline("position")
    rotation = solver.create_spline("rotation")
    
    # Add position channels
    x = position.add_channel("x")
    y = position.add_channel("y", interpolation="linear")
    z = position.add_channel("z", interpolation="step")
    
    # Add rotation channels
    angle = rotation.add_channel("angle", interpolation="cubic")
    torque = rotation.add_channel("torque", interpolation="linear")
    
    # Add keyframes to position.x
    x.add_keyframe(at=0.0, value=0.0)
    x.add_keyframe(at=0.5, value=10.0)
    x.add_keyframe(at=1.0, value=0.0)
    
    # Add keyframes to position.y
    y.add_keyframe(at=0.0, value=5.0)
    y.add_keyframe(at=0.3, value=15.0)
    y.add_keyframe(at=0.7, value=8.0)
    y.add_keyframe(at=1.0, value=5.0)
    
    # Add keyframes to position.z
    z.add_keyframe(at=0.0, value=0.0)
    z.add_keyframe(at=0.25, value=7.0)
    z.add_keyframe(at=0.5, value=-3.0)
    z.add_keyframe(at=0.75, value=4.0)
    z.add_keyframe(at=1.0, value=0.0)
    
    # Add keyframes to rotation.angle
    angle.add_keyframe(at=0.0, value=0.0)
    angle.add_keyframe(at=0.5, value=180.0)
    angle.add_keyframe(at=1.0, value=360.0)
    
    # Add keyframes to rotation.torque
    torque.add_keyframe(at=0.0, value=0.0)
    torque.add_keyframe(at=0.25, value=5.0)
    torque.add_keyframe(at=0.75, value=-5.0)
    torque.add_keyframe(at=1.0, value=0.0)
    
    print("Demo 1: Default (overlay=True) - All channels in one plot")
    print("=======================================================")
    # By default, all channels from all splines are shown in a single plot
    solver.plot(samples=200, theme="light")
    
    print("\nDemo 2: Dark theme with overlay=True")
    print("===================================")
    # Same as above but with dark theme
    solver.plot(samples=200, theme="dark")
    
    print("\nDemo 3: Separate subplots (overlay=False)")
    print("=======================================")
    # Each spline gets its own subplot
    solver.plot(samples=200, overlay=False)
    
    print("\nDemo 4: Dark theme with overlay=False")
    print("===================================")
    # Dark theme with separate subplots
    solver.plot(samples=200, theme="dark", overlay=False)
    
    print("\nDemo 5: Save both versions")
    print("========================")
    # Save both versions to files
    solver.save_plot("overlay_true.png", samples=200, overlay=True)
    print("Saved overlay_true.png")
    
    solver.save_plot("overlay_false.png", samples=200, overlay=False)
    print("Saved overlay_false.png")
    
    print("\nDemo 6: Dark theme save")
    print("=====================")
    # Save dark theme versions
    solver.save_plot("overlay_true_dark.png", samples=200, theme="dark", overlay=True)
    print("Saved overlay_true_dark.png")
    
    solver.save_plot("overlay_false_dark.png", samples=200, theme="dark", overlay=False)
    print("Saved overlay_false_dark.png")

if __name__ == "__main__":
    main()