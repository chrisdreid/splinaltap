#!/usr/bin/env python3
"""
Complex examples showcasing different visualization themes in SplinalTap.
This script generates example plots for the README and unit tests.
"""

from splinaltap.solver import KeyframeSolver
import os
import math

def create_complex_solver(name="ComplexExample"):
    """Create a complex solver with multiple splines and channels for visualization testing."""
    solver = KeyframeSolver(name=name)
    
    # Add built-in variables
    solver.set_variable("pi", math.pi)
    solver.set_variable("amplitude", 5)
    solver.set_variable("frequency", 2)
    
    # Create position spline with multiple channels
    position = solver.create_spline("position")
    x = position.add_channel("x", interpolation="cubic")
    y = position.add_channel("y", interpolation="linear")
    z = position.add_channel("z", interpolation="step")
    
    # Create complex keyframes for position.x using cubic interpolation
    x.add_keyframe(at=0.0, value=0.0)
    x.add_keyframe(at=0.25, value=5.0)  # Use fixed amplitude value
    x.add_keyframe(at=0.5, value=0.0)
    x.add_keyframe(at=0.75, value=-5.0)  # Use fixed amplitude value
    x.add_keyframe(at=1.0, value=0.0)
    
    # Create keyframes for position.y using linear interpolation
    y.add_keyframe(at=0.0, value=0.0)
    y.add_keyframe(at=0.2, value=3.0)
    y.add_keyframe(at=0.4, value=-2.0)
    y.add_keyframe(at=0.6, value=1.0)
    y.add_keyframe(at=0.8, value=-1.0)
    y.add_keyframe(at=1.0, value=0.0)
    
    # Create keyframes for position.z using step interpolation
    z.add_keyframe(at=0.0, value=0.0)
    z.add_keyframe(at=0.2, value=2.0)
    z.add_keyframe(at=0.4, value=-2.0)
    z.add_keyframe(at=0.6, value=1.0)
    z.add_keyframe(at=0.8, value=-1.0)
    z.add_keyframe(at=1.0, value=0.0)
    
    # Create rotation spline
    rotation = solver.create_spline("rotation")
    angle = rotation.add_channel("angle", interpolation="cubic")
    
    # Create complex keyframes for rotation.angle
    angle.add_keyframe(at=0.0, value=0.0)
    angle.add_keyframe(at=0.5, value=180.0)
    angle.add_keyframe(at=1.0, value=360.0)
    
    # Create expression spline with mathematical functions
    expressions = solver.create_spline("expressions")
    
    # Sine wave channel
    sine = expressions.add_channel("sine")
    sine.add_keyframe(at=0.0, value="sin(t * 2 * pi)")
    
    # Cosine wave channel
    cosine = expressions.add_channel("cosine")
    cosine.add_keyframe(at=0.0, value="cos(t * 2 * pi)")
    
    # Complex mathematical curve
    complex_curve = expressions.add_channel("complex")
    complex_curve.add_keyframe(at=0.0, value="sin(t * pi) * cos(t * 2 * pi)")
    
    # Create scaling spline with interdependent channels
    solver.set_publish("position.x", ["*"])  # Globally publish position.x
    
    scaling = solver.create_spline("scaling")
    uniform = scaling.add_channel("uniform")
    uniform.add_keyframe(at=0.0, value=1.0)
    uniform.add_keyframe(at=0.5, value=2.0)
    uniform.add_keyframe(at=1.0, value=1.0)
    
    # Channel that depends on position.x
    dependent = scaling.add_channel("dependent")
    dependent.add_keyframe(at=0.0, value="position.x + 1")
    
    return solver

def generate_theme_examples(output_dir="/home/chris/dev/venv/v-splinaltap/unittest"):
    """Generate example plots with different themes and save to files.
    
    Args:
        output_dir: Directory where example images should be saved
    
    Returns:
        Paths to the generated images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create the example solver
    solver = create_complex_solver()
    
    # Generate overlay plots for all themes
    image_paths = []
    
    # Dark theme (default)
    dark_path = os.path.join(output_dir, "theme_dark.png")
    solver.save_plot(dark_path, samples=200, theme="dark", overlay=True)
    image_paths.append(dark_path)
    
    # Medium theme
    medium_path = os.path.join(output_dir, "theme_medium.png")
    solver.save_plot(medium_path, samples=200, theme="medium", overlay=True)
    image_paths.append(medium_path)
    
    # Light theme
    light_path = os.path.join(output_dir, "theme_light.png")
    solver.save_plot(light_path, samples=200, theme="light", overlay=True)
    image_paths.append(light_path)
    
    # Generate separate plots (non-overlay)
    separated_path = os.path.join(output_dir, "separate_splines.png")
    solver.save_plot(separated_path, samples=200, theme="dark", overlay=False)
    image_paths.append(separated_path)
    
    # Filtered plots (only specific channels)
    filter_channels = {
        "expressions": ["sine", "cosine"],
        "position": ["x"]
    }
    filtered_path = os.path.join(output_dir, "filtered_channels.png")
    solver.save_plot(filtered_path, samples=200, theme="dark", filter_channels=filter_channels)
    image_paths.append(filtered_path)
    
    # Single spline plot (position)
    position = solver.get_spline("position")
    position_path = os.path.join(output_dir, "single_spline.png")
    position.save_plot(position_path, samples=200, theme="dark", title="Position Channels")
    image_paths.append(position_path)
    
    return image_paths

def get_example_cli_command():
    """Return example CLI commands that reproduce the same plots."""
    commands = [
        "# Generate dark theme plot (default)",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize save=theme_dark_cli.png",
        "",
        "# Generate medium theme plot",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize theme=medium save=theme_medium_cli.png",
        "",
        "# Generate light theme plot",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize theme=light save=theme_light_cli.png",
        "",
        "# Generate separated subplots",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize overlay=false save=separate_cli.png"
    ]
    
    return "\n".join(commands)

if __name__ == "__main__":
    # Generate examples and print their paths
    output_paths = generate_theme_examples()
    print("Generated example images:")
    for path in output_paths:
        print(f"- {path}")
        
    # Print CLI commands
    print("\nExample CLI commands:")
    print(get_example_cli_command())