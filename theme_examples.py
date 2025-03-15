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

def generate_theme_examples(output_dir=None):
    """Generate example plots with different themes and save to files.
    
    Args:
        output_dir: Directory where example images should be saved
    
    Returns:
        Paths to the generated images
    """
    # Use unittest/output directory by default
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'unittest', 'output'
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a cleaner, simpler solver for SVG examples
    clean_solver = KeyframeSolver(name="VisualExamples")
    
    # Add variables for expressions
    clean_solver.set_variable("pi", math.pi)
    
    # Create position spline with multiple channels using different interpolation methods
    position = clean_solver.create_spline("position")
    
    # Cubic interpolation channel
    cubic = position.add_channel("cubic", interpolation="cubic")
    cubic.add_keyframe(at=0.0, value=0.0)
    cubic.add_keyframe(at=0.25, value=2.5)
    cubic.add_keyframe(at=0.75, value=-2.5)
    cubic.add_keyframe(at=1.0, value=0.0)
    
    # Linear interpolation channel
    linear = position.add_channel("linear", interpolation="linear")
    linear.add_keyframe(at=0.0, value=0.0)
    linear.add_keyframe(at=0.25, value=2.0)
    linear.add_keyframe(at=0.5, value=0.0)
    linear.add_keyframe(at=0.75, value=-2.0)
    linear.add_keyframe(at=1.0, value=0.0)
    
    # Bezier interpolation with control points
    bezier = position.add_channel("bezier", interpolation="bezier")
    bezier.add_keyframe(at=0.0, value=0.0, control_points=[0.1, 3.0, 0.2, 3.5])
    bezier.add_keyframe(at=0.5, value=0.0, control_points=[0.3, -3.5, 0.4, -3.0])
    bezier.add_keyframe(at=1.0, value=0.0, control_points=[0.6, 3.0, 0.9, 1.0])
    
    # Create expression spline
    expressions = clean_solver.create_spline("expressions")
    
    # Sine wave with expression
    sine = expressions.add_channel("sine", interpolation="cubic")
    sine.add_keyframe(at=0.0, value="sin(t * 2 * pi) * 2.5")
    
    # Hermite interpolation with derivatives
    hermite = expressions.add_channel("hermite", interpolation="hermite")
    hermite.add_keyframe(at=0.0, value=-3.0, derivative=0.0)
    hermite.add_keyframe(at=0.33, value=1.0, derivative=6.0)
    hermite.add_keyframe(at=0.66, value=1.0, derivative=-6.0)
    hermite.add_keyframe(at=1.0, value=-3.0, derivative=0.0)
    
    # Generate high quality SVG images with 300 samples
    image_paths = []
    
    # Dark theme (default)
    dark_path = os.path.join(output_dir, "theme_dark.svg")
    clean_solver.save_plot(dark_path, samples=300, theme="dark", overlay=True)
    image_paths.append(dark_path)
    
    # Medium theme
    medium_path = os.path.join(output_dir, "theme_medium.svg")
    clean_solver.save_plot(medium_path, samples=300, theme="medium", overlay=True)
    image_paths.append(medium_path)
    
    # Light theme
    light_path = os.path.join(output_dir, "theme_light.svg")
    clean_solver.save_plot(light_path, samples=300, theme="light", overlay=True)
    image_paths.append(light_path)
    
    # Generate separate plots (non-overlay)
    separated_path = os.path.join(output_dir, "separate_splines.svg")
    clean_solver.save_plot(separated_path, samples=300, theme="dark", overlay=False)
    image_paths.append(separated_path)
    
    # Filtered plots (only specific channels)
    filter_channels = {
        "position": ["cubic", "bezier"]
    }
    filtered_path = os.path.join(output_dir, "filtered_channels.svg")
    clean_solver.save_plot(filtered_path, samples=300, theme="dark", filter_channels=filter_channels)
    image_paths.append(filtered_path)
    
    # Single spline plot (position)
    position = clean_solver.get_spline("position")
    position_path = os.path.join(output_dir, "single_spline.svg")
    position.save_plot(position_path, samples=300, theme="dark")
    image_paths.append(position_path)
    
    # Create a more visual example specifically for the README
    example_solver = KeyframeSolver(name="ExampleSolver")
    example_solver.set_variable("pi", math.pi)
    
    # Create a main spline for the example
    example = example_solver.create_spline("example")
    
    # Add different interpolation methods to show in example
    cubic = example.add_channel("cubic", interpolation="cubic")
    cubic.add_keyframe(at=0.0, value=0.0)
    cubic.add_keyframe(at=0.3, value=2.0)
    cubic.add_keyframe(at=0.7, value=-1.0)
    cubic.add_keyframe(at=1.0, value=0.0)
    
    bezier = example.add_channel("bezier", interpolation="bezier")
    bezier.add_keyframe(at=0.0, value=-2.0, control_points=[0.1, 0.0, 0.2, 1.0])
    bezier.add_keyframe(at=0.5, value=3.0, control_points=[0.3, 5.0, 0.4, 4.0])
    bezier.add_keyframe(at=1.0, value=-2.0, control_points=[0.6, 1.0, 0.9, -3.0])
    
    # Create a visually appealing example for the README
    example_path = os.path.join(output_dir, "example-01.svg")
    example_solver.save_plot(example_path, samples=300, theme="dark")
    image_paths.append(example_path)
    
    return image_paths

def get_example_cli_command():
    """Return example CLI commands that reproduce the same plots."""
    commands = [
        "# Generate dark theme plot (default)",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize save=theme_dark_cli.svg",
        "",
        "# Generate medium theme plot",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize theme=medium save=theme_medium_cli.svg",
        "",
        "# Generate light theme plot",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize theme=light save=theme_light_cli.svg",
        "",
        "# Generate separated subplots",
        "python -m splinaltap.cli --keyframes \"0:0@cubic\" \"0.25:5@cubic\" \"0.5:0@cubic\" \"0.75:-5@cubic\" \"1:0@cubic\" --samples 200 --visualize overlay=false save=separate_cli.svg"
    ]
    
    return "\n".join(commands)

def create_goes_to_eleven_example(output_dir=None):
    """Create a specialized visualization that 'goes to eleven'.
    
    This creates a beautiful comparison of multiple interpolation methods,
    with one method that stands out by "going to eleven" - a humorous reference
    to the Spinal Tap movie and SplinalTap's name origin.
    
    Args:
        output_dir: Directory where to save the output image
    
    Returns:
        Path to the generated image
    """
    # Use unittest/output directory by default
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'unittest', 'output'
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a special solver for this visualization
    solver = KeyframeSolver(name="GoesToEleven")
    solver.set_variable("pi", math.pi)
    
    # Create a comparison spline
    methods = solver.create_spline("methods")
    
    # Create several standard methods channels
    linear = methods.add_channel("linear", interpolation="linear")
    linear.add_keyframe(at=0.0, value=0.0)
    linear.add_keyframe(at=0.5, value=5.0)
    linear.add_keyframe(at=1.0, value=0.0)
    
    cubic = methods.add_channel("cubic", interpolation="cubic")
    cubic.add_keyframe(at=0.0, value=0.0)
    cubic.add_keyframe(at=0.5, value=5.0)
    cubic.add_keyframe(at=1.0, value=0.0)
    
    hermite = methods.add_channel("hermite", interpolation="hermite")
    hermite.add_keyframe(at=0.0, value=0.0, derivative=0.0)
    hermite.add_keyframe(at=0.5, value=5.0, derivative=0.0)
    hermite.add_keyframe(at=1.0, value=0.0, derivative=0.0)
    
    # Create a channel that "goes to eleven"
    eleven = methods.add_channel("eleven", interpolation="cubic")
    eleven.add_keyframe(at=0.0, value=0.0)
    eleven.add_keyframe(at=0.25, value=3.0)
    eleven.add_keyframe(at=0.4, value=8.0)
    eleven.add_keyframe(at=0.5, value=11.0)  # This one goes to eleven!
    eleven.add_keyframe(at=0.6, value=8.0)
    eleven.add_keyframe(at=0.75, value=3.0)
    eleven.add_keyframe(at=1.0, value=0.0)
    
    # Create a spline for showcasing a beautiful single spline
    showcase = solver.create_spline("showcase")
    
    # Create a beautiful showcase with multiple interpolation methods
    showcase_channel = showcase.add_channel("beautiful", interpolation="cubic")
    showcase_channel.add_keyframe(at=0.0, value=0.0)
    showcase_channel.add_keyframe(at=0.2, value=3.0)
    showcase_channel.add_keyframe(at=0.4, value=-2.0)
    showcase_channel.add_keyframe(at=0.6, value=5.0)
    showcase_channel.add_keyframe(at=0.8, value=-3.0)
    showcase_channel.add_keyframe(at=1.0, value=0.0)
    
    # Save the goes-to-eleven comparison plot
    eleven_path = os.path.join(output_dir, "goes_to_eleven.svg")
    solver.save_plot(eleven_path, samples=500, theme="dark")
    
    # Save the beautiful showcase plot
    showcase_spline = solver.get_spline("showcase")
    showcase_path = os.path.join(output_dir, "beautiful_spline.svg")
    showcase_spline.save_plot(showcase_path, samples=500, theme="dark")
    
    return [eleven_path, showcase_path]

if __name__ == "__main__":
    # Generate examples and print their paths
    output_paths = generate_theme_examples()
    print("Generated example images:")
    for path in output_paths:
        print(f"- {path}")
    
    # Generate "goes to eleven" examples
    eleven_paths = create_goes_to_eleven_example()
    print("\nGenerated 'goes to eleven' examples:")
    for path in eleven_paths:
        print(f"- {path}")
        
    # Print CLI commands
    print("\nExample CLI commands:")
    print(get_example_cli_command())