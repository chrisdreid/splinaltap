# splinaltap

*Keyframe interpolation and expression evaluation that goes to eleven!*

## Introduction

SplinalTap is a Python library for advanced interpolation and curve generation with a focus on scientific and mathematical applications. It provides a flexible architecture for defining, manipulating, and evaluating interpolated values using various mathematical methods.

Key capabilities include:
- Multi-channel interpolation with different methods per channel
- Safe mathematical expression evaluation within keyframes
- Multiple interpolation algorithms (cubic, linear, bezier, hermite, etc.)
- GPU acceleration for processing large datasets
- Comprehensive serialization and deserialization
- Command-line interface for quick data processing
- Visualization tools for analyzing interpolation results

Whether you're working with signal processing, function approximation, numerical analysis, or data visualization, SplinalTap provides the necessary tools to define complex interpolation behaviors with an intuitive API.

## Basic Usage

```python
from splinaltap import KeyframeSolver, Spline, Channel

# Create a keyframe solver
solver = KeyframeSolver(name="Interpolation")

# Create a spline and channel
spline = solver.create_spline("main")
channel = spline.add_channel("value", interpolation="cubic")

# Set keyframes with expressions
channel.add_keyframe(at=0.0, value=0)             # Start at 0
channel.add_keyframe(at=0.5, value="sin(t * pi) * 10")  # Use expression with t variable
channel.add_keyframe(at=1.0, value=0)             # End at 0

# Evaluate at any point
value = channel.get_value(0.25)                  # ≈ 7.07 (using cubic interpolation)

# Evaluate across splines
result = solver.solve(0.5)                       # Get all channel values at position 0.5
```

## Advanced Usage

```python
from splinaltap import KeyframeSolver

# Create a keyframe solver
solver = KeyframeSolver(name="3D Vector Field")

# Set variables for use in expressions
solver.set_variable("amplitude", 10)
solver.set_variable("frequency", 2)
solver.set_variable("pi", 3.14159)

# Create coordinate vector spline with multiple channels
coordinates = solver.create_spline("coordinates")
x_channel = coordinates.add_channel("x", interpolation="linear")
y_channel = coordinates.add_channel("y", interpolation="cubic")
z_channel = coordinates.add_channel("z", interpolation="bezier")

# Add keyframes to each channel
x_channel.add_keyframe(at=0.0, value=0)
x_channel.add_keyframe(at=1.0, value="10 * t")

y_channel.add_keyframe(at=0.0, value=0)
y_channel.add_keyframe(at=0.5, value="amplitude * sin(t * frequency * pi)")
y_channel.add_keyframe(at=1.0, value=0)

z_channel.add_keyframe(at=0.0, value=0, control_points=[0.1, 2, 0.3, 5])
z_channel.add_keyframe(at=1.0, value=0, control_points=[0.7, 5, 0.9, 2])

# Create phase spline
phase = solver.create_spline("phase")
angle = phase.add_channel("angle")
angle.add_keyframe(at=0.0, value=0)
angle.add_keyframe(at=1.0, value=360)

# Create noise spline using random functions
noise = solver.create_spline("noise")
white_noise = noise.add_channel("white")
white_noise.add_keyframe(at=0.0, value="rand() * 2 - 1")  # Random values between -1 and 1
white_noise.add_keyframe(at=1.0, value="rand() * 2 - 1")

quant_noise = noise.add_channel("quantized")
quant_noise.add_keyframe(at=0.0, value="randint([0, 5]) * 0.2")  # Quantized to 0, 0.2, 0.4, 0.6, 0.8, 1.0
quant_noise.add_keyframe(at=1.0, value="randint([0, 5]) * 0.2")

# Save to file
solver.save("parameter_data.json")

# Load from file
loaded = KeyframeSolver.load("parameter_data.json")

# Evaluate at multiple positions
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    result = loaded.solve(t)
    print(f"At {t}: {result}")
```

## About splinaltap

splinaltap is a Python library that provides powerful tools for working with keyframes, expressions, and spline interpolation. It allows you to define keyframes with mathematical expressions, evaluate them at any parametric position along a normalized range, and interpolate between them using various mathematical methods.

### Why the Name?

The name "splinaltap" is a playful nod to the mockumentary "This Is Spinal Tap" and its famous "these go to eleven" scene - because sometimes regular interpolation just isn't enough. But more importantly:

- **splin**: Refers to splines, the mathematical curves used for smooth interpolation
- **al**: Represents algorithms and algebraic expressions
- **tap**: Describes how you can "tap into" the curve at any point to extract values

## Key Features

- 🔢 **Safe Expression Evaluation**: Define keyframes using string expressions that are safely evaluated using Python's AST
- 🔄 **Multiple Interpolation Methods**: Choose from 9 different interpolation algorithms:
  - Nearest Neighbor
  - Linear
  - Polynomial (Lagrange)
  - Quadratic Spline
  - Cubic Spline
  - Hermite Interpolation
  - Bezier Interpolation
  - PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
  - Gaussian Process Interpolation (requires NumPy)
- 🎲 **Random Value Functions**: Generate random values in expressions:
  - `rand()`: Returns a random float between 0 and 1
  - `randint([min, max])`: Returns a random integer between min and max (inclusive)
  - `randint(max)`: Returns a random integer between 0 and max (inclusive)
- 🧮 **Variable Support**: Define and use variables in your expressions for complex mathematical transformations
- 🖥️ **Command Line Interface**: Access all features from the command line

## Command Line Interface

```bash
# Sample at specific points
splinaltap --keyframes "0:0@cubic" "0.5:10@cubic" "1:0@cubic" --samples 0.25 0.5 0.75

# Create a visualization
splinaltap --visualize --keyframes "0:0@cubic" "0.5:sin(t*pi)@cubic" "1:0@cubic" --samples 100

# Use channel-specific methods
splinaltap --keyframes "0:0@linear" "1:10@linear" --samples 0.5@position:linear@rotation:hermite

# Use expressions with variables
splinaltap --keyframes "0:0@cubic" "0.5:amplitude*sin(t*pi)@cubic" "1:0@cubic" --variables "amplitude=10" --samples 100 

# Save and load from files
splinaltap --input-file data.json --samples 100 --output-file output.csv
```
- 🎛️ **Channel Support**: Pass in dynamic channel values that can be used in expressions at runtime
- 🔢 **Multi-component Support**: Interpolate vectors, scalars, and other multi-component values
- 📊 **Visualization**: Built-in support for visualizing interpolation results
- 🔒 **Safe Execution**: No unsafe `eval()` - all expressions are parsed and evaluated securely
- 🚀 **GPU Acceleration**: Optional GPU support via CuPy or JAX for faster processing

## Installation

```bash
pip install splinaltap
```

### Optional Dependencies

For enhanced performance and additional features, you can install these dependencies:

```bash
# For NumPy support (CPU acceleration)
pip install numpy

# For YAML output format support
pip install pyyaml

# For CUDA 11.x GPU support
pip install cupy-cuda11x

# For CUDA 12.x GPU support
pip install cupy-cuda12x

# For JAX support (GPU acceleration with autodiff)
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verifying CUDA Installation

You can verify your CUDA installation is working properly with:

```python
import splinaltap
from splinaltap.backends import BackendManager

# Check available backends
print(BackendManager.available_backends())  # Should include 'cupy' if installed correctly

# Set backend to CuPy
BackendManager.set_backend('cupy')

# Create a sample solver and verify it's using GPU
solver = splinaltap.KeyframeSolver()
spline = solver.create_spline("test")
channel = spline.add_channel("value")
channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=1.0, value=10)

# Generate samples using GPU
samples = [channel.get_value(i/1000) for i in range(1001)]
print(f"Backend used: {BackendManager.get_backend().name}")
print(f"Supports GPU: {BackendManager.get_backend().supports_gpu}")
```

## Quick Start

```python
from splinaltap import KeyframeSolver, Spline, Channel
import matplotlib.pyplot as plt

# Create a solver and spline
solver = KeyframeSolver(name="Interpolation")
spline = solver.create_spline("main")
channel = spline.add_channel("value", interpolation="cubic")

# Add keyframes with expressions
channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=0.25, value="sin(t) + 1")  # 't' is the current position
channel.add_keyframe(at=0.57, value="pow(t, 2)")
channel.add_keyframe(at=1.0, value=10)

# Define a variable
solver.set_variable("amplitude", 2.5)
channel.add_keyframe(at=0.7, value="amplitude * sin(t)")

# Use random functions in expressions
channel.add_keyframe(at=0.9, value="rand() * 5")          # Random float between 0 and 5
channel.add_keyframe(at=0.95, value="randint([1, 10])")   # Random integer between 1 and 10

# Evaluate at various points
positions = [i * 0.01 for i in range(101)]
values = [channel.get_value(p) for p in positions]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(positions, values)
plt.title("Cubic Spline Interpolation")
plt.grid(True)
plt.show()
```

## Command Line Interface

SplinalTap includes a powerful command-line interface for working with interpolation data without writing code.

### Key CLI Principles

SplinalTap follows these consistent principles across all commands:

1. **Default Behavior**: Sampling/evaluation is the default behavior (no command needed)
2. **Normalized 0-1 Range**: By default, all keyframe positions and sample points use a normalized 0-1 range for better numerical precision
3. **Keyframe Syntax**: Use `position:value@method{parameters}` format for direct keyframe definition
4. **Consistent Parameter Names**: 
   - Use `--samples` for specifying sample points
   - Use `--methods` for interpolation methods specification
5. **Channel-Specific Syntax**: Use `@channel:method` syntax for per-channel interpolation
6. **Direct Keyframe Specification**: Define keyframes directly with `--keyframes` without requiring JSON files

### Usage

SplinalTap can be used in two ways, both of which keep all code contained within the git repository:

```bash
# Run from any directory by providing the path (development mode):
python /path/to/splinaltap --help

# If installed with pip (production mode):
python splinaltap --help
```

**IMPORTANT**: All CLI functionality is contained entirely within the `splinaltap` directory. 
This design decision ensures:

1. Repository integrity is maintained
2. All code is properly versioned
3. The package can be installed and run consistently from any location
4. No external scripts or files are needed outside the directory

### Available Commands

The CLI provides several unified commands that follow a consistent pattern. Here are the main commands:

```bash
# Default behavior: sample/evaluate interpolated values (no command needed)
python splinaltap --input-file input.json --samples 0.25 0.5 0.75 --output-file values.csv
python splinaltap --input-file input.json --samples 1000 --range 0,1 --output-file evenly_spaced.csv

# Visualize interpolation
python splinaltap --visualize --input-file input.json --methods cubic --output-file output.png

# Compare multiple interpolation methods (requires --visualize command)
python splinaltap --visualize --input-file input.json --methods linear cubic hermite bezier --compare --output-file comparison.png

# Scene management with unified --scene command
python splinaltap --scene "info scene.json"                         # Show scene info
python splinaltap --scene "ls scene.json"                           # List interpolators
python splinaltap --scene "convert input.json output.yaml"          # Convert formats
python splinaltap --scene "extract scene.json new.json position"    # Extract full interpolator
python splinaltap --scene "extract scene.json pos_x.json position.x" # Extract specific dimension

# Backend management with unified --backend command
python splinaltap --backend                 # Show current backend
python splinaltap --backend ls              # List all backends
python splinaltap --backend info            # Show detailed info
python splinaltap --backend numpy           # Set backend to numpy
python splinaltap --backend best            # Use best available backend
python splinaltap --backend cupy --input-file input.json --samples 100  # Run with cupy backend

# Define and use keyframes directly on command line (0-1 normalized range)
python splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 100 --output-file from_cli.csv

# Use different output formats with --content-type
python splinaltap --keyframes 0:0 0.5:10 1:0 --samples 10 --content-type json
python splinaltap --keyframes 0:0 0.5:10 1:0 --samples 10 --content-type csv --output-file output.csv
python splinaltap --keyframes 0:0 0.5:10 1:0 --samples 10 --content-type yaml
python splinaltap --keyframes 0:0 0.5:10 1:0 --samples 10 --content-type text

# Generate scene files to use as starting points
python splinaltap --generate-scene template.json
python splinaltap --generate-scene my_template.json --keyframes 0:0 0.5:10 1:0
python splinaltap --generate-scene vector_template.json --dimensions 3
python splinaltap --generate-scene template.yaml --content-type yaml

# Work with existing files to create new scenes
python splinaltap --input-file existing.json --generate-scene modified.json
python splinaltap --input-file existing.json --generate-scene with_new_keyframes.json --keyframes 0:0 0.5:5 1:0

# Work with scenes (multiple interpolators)
python splinaltap --scene "info path/to/scene.json"
python splinaltap --scene "ls path/to/scene.json"
python splinaltap --scene "convert input.json output.yaml"
python splinaltap --scene "extract input.json output.json position"         # Extract full position interpolator
python splinaltap --scene "extract input.json position_x.json position.x"   # Extract just the x dimension

# Manage compute backends
python splinaltap --backend                  # Show current backend
python splinaltap --backend ls               # List available backends
python splinaltap --backend info             # Show detailed backend info
python splinaltap --backend numpy            # Switch to NumPy backend
python splinaltap --backend best             # Use best available backend

# Use specific backend for a command
python splinaltap --backend numpy --input-file input.json --samples 0.5 0.75  # Run with numpy backend
python splinaltap --backend cupy --visualize --input-file input.json          # Visualize using cupy
python splinaltap --backend jax --input-file input.json --samples 1000        # Generate 1000 samples with JAX
```

### Input File Format

SplinalTap supports two main JSON file formats: single-dimension interpolators and multi-dimension interpolators.

#### Single-Dimension Interpolator

The simplest format represents a single interpolator:

```json
{
  "version": "1.0",
  "name": "MySolver",
  "range": [0.0, 1.0],
  "variables": {
    "amplitude": 2.5,
    "frequency": 0.5
  },
  "splines": {
    "main": {
      "value": {
        "keyframes": [
          {
            "at": 0.0,
            "value": 0
          },
          {
            "at": 0.5,
            "value": "sin(t*frequency)*amplitude",
            "derivative": 0.5,
            "control_points": [0.6, 12, 0.7, 8]
          },
          {
            "at": 1.0,
            "value": 10
          }
        ]
      }
    }
  }
}
```

#### Multi-Dimension Interpolator

Multi-dimensional data (like positions, colors, etc.) can be organized in a single JSON file with dimensions as properties:

```json
{
  "version": "1.0",
  "name": "MySolver",
  "range": [0.0, 1.0],
  "variables": {
    "amplitude": 2.5,
    "frequency": 0.5
  },
  "splines": {
    "position": {
      "x": {
        "range": [0.0, 1.0],
        "keyframes": [
          { "at": 0.0, "value": 0 },
          { "at": 0.5, "value": "sin(t*frequency)*amplitude" },
          { "at": 1.0, "value": 10 }
        ]
      },
      "y": {
        "variables": {
          "amplitude": 5.0
        },
        "keyframes": [
          { "at": 0.0, "value": 5 },
          { "at": 0.5, "value": 15 },
          { "at": 1.0, "value": 5 }
        ]
      },
      "z": {
        "keyframes": [
          { "at": 0.0, "value": 0 },
          { "at": 1.0, "value": 0 }
        ]
      }
    }
  }
}
```

This structure allows you to:
- Define global range and variables that apply to all dimensions
- Override range or variables at the dimension level if needed
- Keep all dimensions in a single file with clear organization
- Have different keyframes for each dimension while sharing the same normalized position space

Both formats allow for complete control over the interpolator configuration, including variables for expressions, derivatives for Hermite interpolation, and control points for Bezier curves.

### Using Keyframes Directly on the Command Line

SplinalTap allows defining keyframes directly on the command line without needing a JSON file. The CLI currently focuses on single-dimension interpolation for simplicity - for multi-dimensional data or complex scenes, JSON files are recommended.

By default, all positions are normalized to the 0-1 range for better floating-point precision:

```bash
# Define keyframes directly in normalized 0-1 range and sample 100 points
python splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 100 

# Use expressions in keyframes (method is optional, defaults to cubic)
python splinaltap --keyframes "0:0" "0.25:sin(t)" "1:t^2" --samples 100

# Include derivatives for Hermite interpolation
python splinaltap --keyframes "0:0@hermite{deriv=0}" "0.5:10@hermite{deriv=2}" "1:0@hermite{deriv=0}" --samples 100

# Define control points for Bezier interpolation (control points are also in 0-1 space)
python splinaltap --keyframes "0:0@bezier{cp=0.1,0,0.2,3}" "0.5:10@bezier{cp=0.6,12,0.7,8}" "1:0@bezier{cp=0.8,-2,0.9,0}" --samples 100

# Only visualization requires an explicit command
python splinaltap --visualize --keyframes 0:0@cubic 0.3:5@linear 0.7:2@cubic 1:10@cubic --compare

# Use variables in expressions
python splinaltap --keyframes "0:0" "0.5:a*sin(t)" "1:b*t" --variables "a=2.5,b=1.5" --samples 100
```

The keyframe syntax is: `position:value@method{parameters}` where:
- `position` is in the normalized 0-1 range by default
- `value` can be a number or expression in quotes
- `@method` specifies the interpolation method (cubic, hermite, bezier, etc.)
- `{parameters}` are optional method-specific parameters:
  - For hermite: `{deriv=value}` - specifies the derivative at this point
  - For bezier: `{cp=x1,y1,x2,y2}` - specifies the control points

If you omit the method or parameters, reasonable defaults will be used. For example:
- `0.5:10` - uses the default method (cubic) with default parameters
- `0.5:10@hermite` - uses hermite method with default derivative (0)

#### Normalized vs. Index Positions

All keyframe positions are normalized to a 0-1 range by default. This provides several benefits:
- Better floating-point precision for animations
- Consistent representation regardless of time scale
- Easier to transfer animations between contexts
- Reduced rounding errors during interpolation

If you need to use absolute indices instead of normalized positions, use the `--use-indices` flag:

```bash
# Use absolute indices rather than normalized 0-1 positions
python splinaltap --keyframes 0:0@cubic 5:10@cubic 10:0@cubic --use-indices --samples 100
```

#### Default Behavior and Commands

Sampling/evaluation is the default behavior when no specific command is provided:

```bash
# These all sample/evaluate interpolated values (default behavior)
python splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 0.25 0.5 0.75
python splinaltap --input-file input.json --samples 100
python splinaltap --keyframes 0:0@cubic 0.5:10@bezier{cp=0.6,12,0.7,8} 1:0@cubic --samples 1000 --range 0,1
```

Specialized operations require explicit commands:
- `--visualize`: For generating plots and visualizations
- `--scene`: Unified command for scene operations with subcommands like `info`, `ls`, `convert`, `extract`
- `--backend`: Unified command for managing compute backends:
  - Used alone: `--backend` (shows current), `--backend ls` (lists all), etc.
  - Used with other commands: `--backend numpy --input-file input.json` (runs with specified backend)
- `--generate-scene`: For creating scene files at a specified filepath

### Advanced Sample Syntax

SplinalTap supports advanced syntax for sample points with channel-specific methods using the `@` notation:

```bash
# Sample with specific channels and methods (sample points in 0-1 normalized range)
python splinaltap --sample --input-file input.json --samples 0.5@channel-a:linear@channel-b:cubic:hermite

# Sample count with multiple methods for a channel
python splinaltap --sample --keyframes 0:0 0.5:10 1:0 --samples 100 @default:linear:cubic:hermite

# Combine sample points with different methods
python splinaltap --sample --keyframes 0:0 0.5:10 1:0 --samples 0.25@x:linear 0.5@x:cubic 0.75@x:hermite

# Mix sample count and channels
python splinaltap --sample --input-file input.json --samples 1000 @position:linear @rotation:hermite
```

The general syntax format is: `VALUE[@CHANNEL:METHOD1:METHOD2...][@CHANNEL:METHOD...]`

Just like keyframe positions, all sample points are normalized to the 0-1 range by default for consistency. When using `--use-indices`, both keyframes and sample points will use absolute index values instead.

#### Relationship Between Samples and Keyframes

The sample parameter (`--samples`) serves two distinct purposes:
1. When given an integer, it generates that many evenly distributed samples across the 0-1 range
2. When given decimal values (like 0.25, 0.5, 0.75), it samples at those specific normalized positions

This flexibility allows both exact point evaluation and dense sampling from a single parameter, maintaining CLI consistency.

### Available Interpolation Methods

SplinalTap supports multiple interpolation methods that can be specified with the `--methods` parameter:

| Method | Description | Required Keyframe Data | Best Use Case |
|--------|-------------|------------------------|---------------|
| `nearest` | Nearest neighbor interpolation | Value only | Step functions, discrete states |
| `linear` | Linear interpolation between points | Value only | Simple transitions, straight lines |
| `quadratic` | Quadratic spline interpolation | Value only | Smooth curves with C¹ continuity |
| `cubic` | Cubic spline interpolation | Value only | Smooth curves with C² continuity (default) |
| `hermite` | Hermite interpolation | Value + derivative | When you need control over tangents |
| `bezier` | Bezier curve interpolation | Value + control points | When you need precise curve shape control |
| `lagrange` | Lagrange polynomial interpolation | Value only | When curve must pass exactly through all points |
| `pchip` | Piecewise Cubic Hermite Interpolating Polynomial | Value only | When you need monotonicity preservation |
| `gaussian` | Gaussian process interpolation | Value only | Statistical interpolation with uncertainty |

All methods can be used with either normalized or index mode. Each method has different characteristics that make it suitable for different applications:
- `cubic` is a good default choice for smooth animations
- `hermite` gives more control by specifying derivatives at keyframes
- `bezier` provides the most control with additional control points
- `nearest` and `linear` are computationally efficient for large datasets

### Variables vs. Channels

SplinalTap has two distinct concepts for parameterizing expressions:

1. **Variables**: Constants defined at creation time, used in expressions for all evaluations
   - Set with `--variables` parameter in CLI or in the input file
   - Examples: `amplitude=2.5`, `frequency=0.5`, `phase=1.2`
   - Cannot be changed after creation without recreating the interpolator

2. **Channels**: Dynamic values passed at evaluation time, changing the result on-the-fly
   - Specified with `--channels` parameter or using the `@channel` syntax
   - Examples: `@position:linear`, `@x:cubic`, `@rotation:hermite`
   - Can be different for each evaluation/sample without recreating anything
   - Useful for dynamic parameters that change over time

Variables are baked into the interpolator at creation, while channels are external inputs that can change with each evaluation.

### Working with Solver Files

A Solver is a collection of multiple named splines, which can be useful for complex datasets with multiple parameters. Solver files extend the multi-component concept to allow completely independent splines:

```json
{
  "version": "1.0",
  "name": "MySolver",
  "metadata": {
    "description": "A complex parameter set with multiple components",
    "author": "SplinalTap User",
    "created": "2023-09-15"
  },
  "range": [0.0, 1.0],  // Global range for all splines
  "variables": {
    "pi": 3.14159,
    "amplitude": 10
  },
  "splines": {
    "position": {  // Spline
      "x": {       // Channel
        "min-max": [0, 10],  // Inferred from values
        "interpolation_method": "cubic",
        "keyframes": [
          {"at": 0.0, "value": 0},
          {"at": 0.5, "value": 10},
          {"at": 1.0, "value": 0}
        ]
      },
      "y": {       // Channel
        "interpolation_method": "cubic",
        "keyframes": [
          {"at": 0.0, "value": 0},
          {"at": 0.5, "value": "sin(t*pi)"},
          {"at": 1.0, "value": 0}
        ]
      },
      "z": {       // Channel
        "interpolation_method": "cubic",
        "keyframes": [
          {"at": 0.0, "value": 0},
          {"at": 1.0, "value": 5}
        ]
      }
    },
    "rotation": {  // Spline
      "interpolation_method": "linear",  // Single value, linear makes sense
      "keyframes": [
        {"at": 0.0, "value": 0},
        {"at": 1.0, "value": 360}
      ]
    },
    "scale": {     // Spline
      "x": {       // Channel
        "min-max": [1, 1],  // Adjusted based on context
        "interpolation_method": "cubic",
        "keyframes": [
          {"at": 0.0, "value": 1},
          {"at": 0.5, "value": "amplitude * 0.1"},  // Evaluates to 1
          {"at": 1.0, "value": 1}
        ]
      },
      "y": {       // Channel
        "min-max": [1, 1],
        "interpolation_method": "cubic",
        "keyframes": [
          {"at": 0.0, "value": 1},
          {"at": 1.0, "value": 1}
        ]
      }
    }
  }
}
```

Scene files can be:
- Inspected with `--scene-info`
- Converted to other formats with `--scene-convert`
- Individual interpolators can be extracted with `--scene-extract`

**Note**: While the CLI supports basic operations with scene files and single-dimension interpolators, complex multi-dimensional configurations are best created and managed through JSON files directly. This design choice keeps the command-line interface focused and intuitive while still providing full power through file-based workflows.

### Output Format

When using SplinalTap to evaluate or sample keyframes, the output follows a simple, consistent format:

```json
{
  "version": "1.0",
  "samples": [0.0, 0.25, 0.5, 0.75, 1.0],
  "results": {
    "chan-x": [0.0, 2.5, 5.0, 7.5, 10.0],
    "position": [0.0, 2.5, 5.0, 7.5, 10.0]
  }
}
```

The output consists of:
- `version`: The format version for compatibility tracking
- `samples`: Array of sample point positions
- `results`: Object containing channels, each with an array of values that directly correspond to the samples

Each channel's values are stored as a simple array, with positions corresponding to the same index in the samples array. This makes the output easy to parse and use in any application.

For more details on each command, run `splinaltap <command> --help`.

## Advanced Usage

### Using Different Interpolation Methods

```python
# Compare different interpolation methods
from splinaltap import KeyframeSolver, Spline, Channel
import matplotlib.pyplot as plt

# Create a solver with a single channel
solver = KeyframeSolver()
spline = solver.create_spline("test")
channel = spline.add_channel("value")

# Add some keyframes
channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=0.5, value=10)
channel.add_keyframe(at=1.0, value=0)

# Compare different interpolation methods
methods = ["linear", "cubic", "hermite", "bezier"]
plt.figure(figsize=(12, 8))

positions = [i * 0.01 for i in range(101)]  # Normalized 0-1 range
for method in methods:
    # Create a channel for each method
    method_channel = spline.add_channel(f"value_{method}", interpolation=method)
    method_channel.add_keyframe(at=0.0, value=0)
    method_channel.add_keyframe(at=0.5, value=10)
    method_channel.add_keyframe(at=1.0, value=0)
    
    values = [method_channel.get_value(p) for p in positions]
    plt.plot(positions, values, label=method.capitalize())

plt.legend()
plt.title("Interpolation Methods Comparison")
plt.show()
```

### Understanding Solvers, Splines, and Channels

SplinalTap works with three core concepts that provide different levels of organization and flexibility:

#### 1. Splines: Independent Interpolation Functions

Splines are named component groups that represent a complete interpolation entity. In a solver file, 
these are named entities (like "coordinates", "phase", "magnitude") that can be manipulated independently.

```python
# A solver can contain multiple independent splines
solver = {
    "splines": {
        "coordinates": { /* coordinates spline data with channels */ },
        "phase": { /* phase spline data with channels */ },
        "magnitude": { /* magnitude spline data with channels */ }
    }
}

# Each spline can be extracted and used independently
coordinates_spline = solver.get_spline("coordinates")
phase_spline = solver.get_spline("phase")
```

When using the `--scene extract` command, you're extracting a named spline from a solver file:
```bash
# Extract the "coordinates" spline including all its channels
python splinaltap --scene "extract scene.json coordinates.json coordinates"
```

#### 2. Channels: Components of a Spline

Channels represent individual components of a spline (like x, y, z components of a vector). 
Each channel has its own set of keyframes and interpolation method but shares the same normalized parametric range.

```python
# Create a 3D vector spline with x, y, z channels
spline = Spline()
spline.add_channel("x")
spline.add_channel("y")
spline.add_channel("z")

# Set keyframes for each channel
spline.channels["x"].add_keyframe(at=0.0, value=0)
spline.channels["x"].add_keyframe(at=1.0, value=10)

spline.channels["y"].add_keyframe(at=0.0, value=0)
spline.channels["y"].add_keyframe(at=1.0, value=20)

spline.channels["z"].add_keyframe(at=0.0, value=0)
spline.channels["z"].add_keyframe(at=1.0, value=5)

# Get the interpolated vector at position 0.25
values = spline.get_value(0.25)  # Returns {"x": 2.5, "y": 5.0, "z": 1.25}

# Access a specific channel
x_value = spline.get_channel_value("x", 0.25)  # Returns 2.5
```

You can extract a specific channel from a spline using the dot notation:
```bash
# Extract just the x channel from the coordinates spline
python splinaltap --scene "extract scene.json coordinates_x.json coordinates.x"
```

#### 3. External Channels vs. Variables

SplinalTap has two distinct ways to parameterize expressions:

1. **Variables**: Constants defined at creation time, baked into expressions for all evaluations
   ```python
   # Set a variable that can be used in keyframe expressions
   solver.set_variable("amplitude", 2.5)
   
   # Use in keyframe expressions
   channel.add_keyframe(at=0.5, value="sin(t) * amplitude")
   ```

2. **External Channels**: Dynamic values passed at evaluation time to influence expressions
   ```python
   # Define keyframes that use external channel values
   channel.add_keyframe(at=0.5, value="a * sin(t) + b")  # Uses channels 'a' and 'b'
   
   # Evaluate with different channel values
   ext_channels = {"a": 1.0, "b": 0.5}  # External parameters
   value = channel.get_value(0.5, ext_channels)
   ```

**Key Differences Summary**: 

- **Splines** are complete, named interpolation functions (coordinates, phase, etc.)
- **Channels** are components of a spline (x, y, z components) with their own keyframes and interpolation methods
- **External channels** are dynamic inputs passed at evaluation time to parameterize expressions
- **Variables** are constants defined at creation time and baked into expressions

**Hierarchy**:
```
Solver
 ├─ Spline: "coordinates" (a vector quantity)
 │   ├─ Channel: "x" (component with its own keyframes and interpolation)
 │   ├─ Channel: "y" (component with its own keyframes and interpolation)
 │   └─ Channel: "z" (component with its own keyframes and interpolation)
 │
 ├─ Spline: "phase" (a scalar quantity)
 │   └─ Channel: "angle" (component with its own keyframes and interpolation)
 │
 └─ Spline: "magnitude" (a multi-component quantity)
     ├─ Channel: "x" (component with its own keyframes and interpolation)
     └─ Channel: "y" (component with its own keyframes and interpolation)
```

### Using Control Points (Bezier)

```python
from splinaltap import KeyframeSolver, Spline, Channel

# Create a solver with a bezier channel
solver = KeyframeSolver()
spline = solver.create_spline("test")
channel = spline.add_channel("value", interpolation="bezier")

# Add keyframe with control points for Bezier interpolation
channel.add_keyframe(at=0.0, value=0, control_points=[0.1, 2, 0.3, 5])
channel.add_keyframe(at=0.4, value=5.0, control_points=[0.42, 6.0, 0.48, 7.0])
channel.add_keyframe(at=1.0, value=0, control_points=[0.7, 5, 0.9, 2])
```

### Using GPU Acceleration

```python
from splinaltap import KeyframeSolver, Spline, Channel
from splinaltap.backends import BackendManager

# Let splinaltap choose the best backend for your workload
BackendManager.use_best_available(data_size=1_000_000, method="cubic")
print(f"Selected backend: {BackendManager.get_backend().name}")

# Create solver and keyframes
solver = KeyframeSolver()  # Default normalized 0-1 range
spline = solver.create_spline("test")
channel = spline.add_channel("value", interpolation="cubic")
channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=1.0, value=10)

# Generate 1 million samples efficiently using the best available backend
samples = [channel.get_value(i/1_000_000) for i in range(1_000_001)]
```

### Exporting to Shader Code

Splinaltap can export your keyframe interpolation as shader code for use in graphics applications:

```python
from splinaltap import KeyframeSolver, Spline, Channel

# Create solver with keyframes
solver = KeyframeSolver()
spline = solver.create_spline("test")
channel = spline.add_channel("value")
channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=0.25, value=5)
channel.add_keyframe(at=0.75, value=2)
channel.add_keyframe(at=1.0, value=10)

# Export as GLSL shader function
glsl_code = channel.export_function(language="glsl", method="linear")
print(glsl_code)

# Export as C/C++ function
c_code = channel.export_function(language="c", method="linear")
print(c_code)
```

Example GLSL output:
```glsl
// GLSL linear interpolation function for 4 keyframes
float linearInterpolate(float t) {
    // Keyframe positions
    float positions[4] = float[4](0.000000, 0.250000, 0.750000, 1.000000);
    // Keyframe values
    float values[4] = float[4](0.000000, 5.000000, 2.000000, 10.000000);

    // Handle out-of-range positions
    if (t <= positions[0]) return values[0];
    if (t >= positions[3]) return values[3];

    // Find the bracketing keyframes
    for (int i = 0; i < positions.length() - 1; i++) {
        if (positions[i] <= t && t <= positions[i + 1]) {
            float alpha = (t - positions[i]) / (positions[i + 1] - positions[i]);
            return mix(values[i], values[i + 1], alpha);
        }
    }

    // Fallback (should never reach here)
    return values[0];
}
```

## Random Functions

SplinalTap provides built-in random value generation in expressions through two functions:

```python
# Random float between 0 and 1
channel.add_keyframe(at=0.5, value="rand()")

# Random integer between min and max (inclusive)
channel.add_keyframe(at=0.5, value="randint([1, 10])")

# Random integer between 0 and max (inclusive)
channel.add_keyframe(at=0.5, value="randint(5)")
```

These functions are evaluated each time `get_value()` is called, generating new random values with each evaluation. This makes them ideal for:

1. **Noise Generation**: Create random noise patterns by evaluating at many points
   ```python
   noise_channel.add_keyframe(at=0.0, value="rand() * 2 - 1")  # Random values from -1 to 1
   ```

2. **Quantized Effects**: Create stepped or quantized random values
   ```python
   # Random values of 0, 0.2, 0.4, 0.6, 0.8, or 1.0
   quant_channel.add_keyframe(at=0.0, value="randint([0, 5]) * 0.2")
   ```

3. **Variation**: Add controlled randomness to otherwise deterministic animations
   ```python
   # Base value with random variation
   channel.add_keyframe(at=0.5, value="sin(t*pi) + rand() * 0.1")  # Add jitter
   ```

4. **Random Selection**: Choose randomly between discrete options
   ```python
   # Select from 0, 5, or 10
   channel.add_keyframe(at=0.5, value="randint([0, 2]) * 5")
   ```

Note that these functions generate new values each time they're evaluated, which means:
- The same position may return different values in successive calls
- Channels with random functions will appear different each time they're sampled
- For consistent random values, you would need to set a seed outside of SplinalTap

## Performance Considerations

Splinaltap uses a tiered approach to performance optimization, automatically choosing the best available implementation based on your hardware and installed packages:

1. **CUDA/CuPy kernels**: Fastest option for large-scale processing on NVIDIA GPUs
2. **JAX/Numba JIT compilation**: Fast GPU and CPU acceleration with automatic optimization
3. **Vectorized NumPy/CuPy**: Efficient batch operations for moderate-sized datasets
4. **Pure Python**: Universal fallback that works everywhere

### CPU vs GPU Performance Tradeoffs

| Operation | CPU Better Than GPU | GPU Better Than CPU |
|-----------|---------------------|---------------------|
| Linear interpolation | < 50,000 points | > 100,000 points |
| Cubic/complex interpolation | < 10,000 points | > 20,000 points |
| Multiple components | < 1,000 components | > 1,000 components |
| Interactive computation | Real-time adjustments | Batch processing |
| Small datasets | Few sample points | Large parallel computations |

### GPU Overhead Considerations

When using GPU acceleration, be aware of these overhead costs:
- PCIe data transfer: ~1-2ms base latency
- GPU memory allocation: ~0.5-1ms
- Kernel launch: ~5-10μs per call

For maximum performance with GPU acceleration:
1. Process data in large batches when possible
2. Keep data on the GPU if it will be used for further processing
3. For interactive applications, consider CPU acceleration with Numba for lower latency

```python
# Example: Choosing the optimal backend automatically
from splinaltap import KeyframeSolver, Spline, Channel
from splinaltap.backends import BackendManager

# Let splinaltap choose the best backend for your system
BackendManager.use_best_available()

# Create a complex interpolation
solver = KeyframeSolver()  # Normalized 0-1 range
spline = solver.create_spline("main")
channel = spline.add_channel("value", interpolation="cubic")

channel.add_keyframe(at=0.0, value=0)
channel.add_keyframe(at=0.25, value="sin(t) + 1")
channel.add_keyframe(at=0.5, value="t^2")
channel.add_keyframe(at=1.0, value=10)

# Generate samples with optimal performance
samples = [channel.get_value(i/100000) for i in range(100001)]
```

## Applications

- Scientific data interpolation
- Parameter optimization
- Function approximation
- Signal processing
- Statistical modeling
- Numerical analysis
- Data visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
