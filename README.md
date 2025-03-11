# splinaltap

*Keyframe interpolation and expression evaluation that goes to eleven!*

## About splinaltap

splinaltap is a Python library that provides powerful tools for working with keyframes, expressions, and spline interpolation. It allows you to define keyframes with mathematical expressions, evaluate them at any point along a timeline, and interpolate between them using various methods.

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
- 🧮 **Variable Support**: Define and use variables in your expressions for complex animations and simulations
- 🎛️ **Channel Support**: Pass in dynamic channel values that can be used in expressions at runtime
- 🔢 **Multi-dimensional Support** *(coming soon)*: Interpolate vectors, colors, and other multi-component values
- 📊 **Visualization**: Built-in support for visualizing interpolation results
- 🔒 **Safe Execution**: No unsafe `eval()` - all expressions are parsed and evaluated securely
- 🚀 **GPU Acceleration**: Optional GPU support via CuPy or JAX for faster processing

## Installation

```bash
pip install splinaltap
```

### Optional Dependencies

For enhanced performance, you can install NumPy (CPU acceleration) or GPU acceleration libraries:

```bash
# For NumPy support (CPU acceleration)
pip install numpy

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

# Create a sample and verify it's using GPU
interpolator = splinaltap.KeyframeInterpolator(10)
interpolator.set_keyframe(0.0, 0)
interpolator.set_keyframe(10.0, 10)

# Generate samples using GPU
samples = interpolator.sample_with_gpu(0, 10, 1000)
print(f"Backend used: {BackendManager.get_backend().name}")
print(f"Supports GPU: {BackendManager.get_backend().supports_gpu}")
```

## Quick Start

```python
from splinaltap import KeyframeInterpolator
import matplotlib.pyplot as plt

# Create a KeyframeInterpolator instance (normalized 0-1 range by default)
interpolator = KeyframeInterpolator()

# Add keyframes with expressions
interpolator.set_keyframe(0.0, 0)
interpolator.set_keyframe(0.25, "sin(t) + 1")  # 't' is the current position
interpolator.set_keyframe(0.57, "pow(t, 2)")
interpolator.set_keyframe(1.0, 10)

# Define a variable
interpolator.set_variable("amplitude", 2.5)
interpolator.set_keyframe(0.7, "amplitude * sin(t)")

# Evaluate at various points
positions = [i * 0.01 for i in range(101)]
values = [interpolator.get_value(p, "cubic") for p in positions]

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
2. **Normalized 0-1 Range**: By default, all keyframe positions and sample points use a normalized 0-1 range for better precision
3. **Keyframe Syntax**: Use `position:value@method{parameters}` format for direct keyframe definition
4. **Consistent Parameter Names**: 
   - Always use `--samples` for specifying sample points (never alternatives like `--points`)
   - Always use `--methods` (plural) for interpolation methods (never singular `--method`)
5. **Channel-Specific Syntax**: Use `@channel:method` syntax for per-channel interpolation
6. **Direct Keyframe Specification**: Define keyframes directly with `--keyframes` without requiring JSON files

### Usage

SplinalTap can be used in two ways, both of which keep all code contained within the git repository:

```bash
# Run from any directory by providing the path (development mode):
python /path/to/splinaltap --help

# If installed with pip (production mode):
splinaltap --help
```

**IMPORTANT**: All CLI functionality is contained entirely within the `splinaltap` directory. 
This design decision ensures:

1. Repository integrity is maintained
2. All code is properly versioned
3. The package can be installed and run consistently from any location
4. No external scripts or files are needed outside the directory

### Available Commands

```bash
# Visualize interpolation
splinaltap --visualize --input-file input.json --methods cubic --output-file output.png

# Compare multiple interpolation methods (requires --visualize command)
splinaltap --visualize --input-file input.json --methods linear cubic hermite bezier --compare --output-file comparison.png

# Default behavior: sample/evaluate interpolated values (no command needed)
splinaltap --input-file input.json --samples 0.25 0.5 0.75 --output-file values.csv
splinaltap --input-file input.json --samples 1000 --range 0,1 --output-file evenly_spaced.csv

# Define and use keyframes directly on command line (0-1 normalized range)
splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 100 --output-file from_cli.csv

# Work with scenes (multiple interpolators)
splinaltap --scene-info --input-file scene.json
splinaltap --scene-convert --input-file scene.json --output-file scene.yaml
splinaltap --scene-extract --input-file scene.json --interpolator-name position_x --output-file extracted.json

# Manage compute backends
splinaltap --backend --list
splinaltap --backend --use-backend numpy
splinaltap --backend --use-best
```

### Input File Format

SplinalTap supports two main JSON file formats: single-dimension interpolators and multi-dimension interpolators.

#### Single-Dimension Interpolator

The simplest format represents a single interpolator:

```json
{
  "range": [0.0, 1.0],          // Optional: Range as [min, max] (default is normalized 0-1)
  "variables": {                // Optional: Constants used in expressions
    "amplitude": 2.5,
    "frequency": 0.5
  },
  "keyframes": [                // Required: Array of keyframes
    {
      "position": 0.0,          // Position (normalized 0-1 by default)
      "value": 0                // Value or expression (number or string)
    },
    {
      "position": 0.5,
      "value": "sin(t * frequency) * amplitude",
      "derivative": 0.5,        // Optional: Derivative for Hermite interpolation
      "control_points": [0.6, 12, 0.7, 8]  // Optional: Control points for Bezier
    },
    {
      "position": 1.0,
      "value": 10
    }
  ]
}
```

#### Multi-Dimension Interpolator

Multi-dimensional data (like positions, colors, etc.) can be organized in a single JSON file with dimensions as properties:

```json
{
  "range": [0.0, 1.0],          // Global range (applies to all dimensions unless overridden)
  "variables": {                // Global variables (accessible to all dimensions)
    "amplitude": 2.5,
    "frequency": 0.5
  },
  "dimensions": {               // Contains all dimensions
    "x": {                      // First dimension (x coordinate)
      "range": [0.0, 1.0],      // Optional: Override global range for this dimension
      "keyframes": [
        { "position": 0.0, "value": 0 },
        { "position": 0.5, "value": "sin(t * frequency) * amplitude" },
        { "position": 1.0, "value": 10 }
      ]
    },
    "y": {                      // Second dimension (y coordinate)
      "keyframes": [            // Uses global range since not overridden
        { "position": 0.0, "value": 5 },
        { "position": 0.5, "value": 15 },
        { "position": 1.0, "value": 5 }
      ],
      "variables": {            // Dimension-specific variables (override globals)
        "amplitude": 5.0
      }
    },
    "z": {                      // Third dimension (z coordinate)
      "keyframes": [
        { "position": 0.0, "value": 0 },
        { "position": 1.0, "value": 0 }
      ]
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
splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 100 

# Use expressions in keyframes (method is optional, defaults to cubic)
splinaltap --keyframes "0:0" "0.25:sin(t)" "1:t^2" --samples 100

# Include derivatives for Hermite interpolation
splinaltap --keyframes "0:0@hermite{deriv=0}" "0.5:10@hermite{deriv=2}" "1:0@hermite{deriv=0}" --samples 100

# Define control points for Bezier interpolation (control points are also in 0-1 space)
splinaltap --keyframes "0:0@bezier{cp=0.1,0,0.2,3}" "0.5:10@bezier{cp=0.6,12,0.7,8}" "1:0@bezier{cp=0.8,-2,0.9,0}" --samples 100

# Only visualization requires an explicit command
splinaltap --visualize --keyframes 0:0@cubic 0.3:5@linear 0.7:2@cubic 1:10@cubic --compare

# Use variables in expressions
splinaltap --keyframes "0:0" "0.5:a*sin(t)" "1:b*t" --variables "a=2.5,b=1.5" --samples 100
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
splinaltap --keyframes 0:0@cubic 5:10@cubic 10:0@cubic --use-indices --samples 100
```

#### Default Behavior and Commands

Sampling/evaluation is the default behavior when no specific command is provided:

```bash
# These all sample/evaluate interpolated values (default behavior)
splinaltap --keyframes 0:0@cubic 0.5:10@cubic 1:0@cubic --samples 0.25 0.5 0.75
splinaltap --input-file input.json --samples 100
splinaltap --keyframes 0:0@cubic 0.5:10@bezier{cp=0.6,12,0.7,8} 1:0@cubic --samples 1000 --range 0,1
```

Specialized operations require explicit commands:
- `--visualize`: For generating plots and visualizations
- `--scene-info`, `--scene-convert`, `--scene-extract`: For scene operations
- `--backend`: For managing compute backends

### Advanced Sample Syntax

SplinalTap supports advanced syntax for sample points with channel-specific methods using the `@` notation:

```bash
# Sample with specific channels and methods (sample points in 0-1 normalized range)
splinaltap --sample --input-file input.json --samples 0.5@channel-a:linear@channel-b:cubic:hermite

# Sample count with multiple methods for a channel
splinaltap --sample --keyframes 0:0 0.5:10 1:0 --samples 100 @default:linear:cubic:hermite

# Combine sample points with different methods
splinaltap --sample --keyframes 0:0 0.5:10 1:0 --samples 0.25@x:linear 0.5@x:cubic 0.75@x:hermite

# Mix sample count and channels
splinaltap --sample --input-file input.json --samples 1000 @position:linear @rotation:hermite
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

### Working with Scene Files

A Scene is a collection of multiple named interpolators, which can be useful for complex animations with multiple properties. Scene files extend the multi-dimension concept to allow completely independent interpolators:

```json
{
  "name": "MyAnimation",
  "metadata": {
    "description": "A complex animation with multiple properties",
    "author": "SplinalTap User",
    "created": "2023-09-15"
  },
  "variables": {                 // Global variables shared by all interpolators
    "pi": 3.14159,
    "amplitude": 10
  },
  "interpolators": {
    "position": {                // A multi-dimensional interpolator for position
      "range": [0.0, 1.0],       // Global range for this interpolator
      "dimensions": {
        "x": {
          "keyframes": [
            {"position": 0.0, "value": 0},
            {"position": 0.5, "value": 10},
            {"position": 1.0, "value": 0}
          ]
        },
        "y": {
          "keyframes": [
            {"position": 0.0, "value": 0},
            {"position": 0.5, "value": "sin(t*pi)"},
            {"position": 1.0, "value": 0}
          ]
        },
        "z": {
          "keyframes": [
            {"position": 0.0, "value": 0},
            {"position": 1.0, "value": 5}
          ]
        }
      }
    },
    "rotation": {               // Single-dimension interpolator for rotation
      "range": [0.0, 1.0],
      "keyframes": [
        {"position": 0.0, "value": 0},
        {"position": 1.0, "value": 360}
      ]
    },
    "scale": {                  // Another multi-dimensional interpolator
      "dimensions": {
        "x": {
          "keyframes": [
            {"position": 0.0, "value": 1},
            {"position": 0.5, "value": "amplitude * 0.1"},
            {"position": 1.0, "value": 1}
          ]
        },
        "y": {
          "keyframes": [
            {"position": 0.0, "value": 1},
            {"position": 1.0, "value": 1}
          ]
        }
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

For more details on each command, run `splinaltap <command> --help`.

## Advanced Usage

### Using Different Interpolation Methods

```python
# Compare different interpolation methods
methods = ["linear", "cubic", "hermite", "bezier"]
plt.figure(figsize=(12, 8))

positions = [i * 0.01 for i in range(101)]  # Normalized 0-1 range
for method in methods:
    values = [interpolator.get_value(p, method) for p in positions]
    plt.plot(positions, values, label=method.capitalize())

plt.legend()
plt.title("Interpolation Methods Comparison")
plt.show()
```

### Understanding Dimensions vs Channels

SplinalTap works with two important concepts that provide different types of flexibility:

#### Dimensions: Components of Your Interpolated Values

Dimensions represent the different components of the values being interpolated (like x, y, z coordinates). 
Each dimension has its own set of keyframes but shares the same normalized range.

```python
# Coming soon: Multi-dimensional support for vectors, colors, and more
# Create a 3D interpolator (x, y, z)
interpolator = KeyframeInterpolator(dimensions=3)

# Set a keyframe with all three dimensions
interpolator.set_keyframe(0.0, [0, 0, 0])  # Start at origin
interpolator.set_keyframe(0.5, [10, 20, 5])  # Mid-point
interpolator.set_keyframe(1.0, [20, 0, 10])  # End point

# Get the interpolated 3D position at position 0.25
position = interpolator.get_value(0.25)  # Returns something like [5, 10, 2.5]
```

#### Channels: Dynamic Parameters for Expressions

Channels are external parameters that you can pass to influence expressions at evaluation time, 
without changing the keyframes themselves. Think of them as knobs you can adjust on-the-fly.

```python
# Define keyframes that use channel values
interpolator.set_keyframe(0.3, "a * sin(t) + b")  # Expression uses channels 'a' and 'b'

# Evaluate with different channel values
channels_1 = {"a": 1.0, "b": 0.5}  # First configuration
channels_2 = {"a": 2.0, "b": 0.0}  # Second configuration

# Same keyframes, different results based on channel values
positions = [i * 0.01 for i in range(101)]
values_1 = [interpolator.get_value(p, "cubic", channels_1) for p in positions]
values_2 = [interpolator.get_value(p, "cubic", channels_2) for p in positions]
```

**Key Difference**: 
- Dimensions are the components being interpolated (outputs)
- Channels are dynamic inputs used at evaluation time
- Variables are constants defined at creation time

### Using Control Points (Bezier)

```python
# Set keyframe with control points for Bezier interpolation
interpolator.set_keyframe(0.4, 5.0, derivative=None, control_points=(0.42, 6.0, 0.48, 7.0))
```

### Using GPU Acceleration

```python
from splinaltap import KeyframeInterpolator
from splinaltap.backends import BackendManager

# Let splinaltap choose the best backend for your workload
BackendManager.use_best_available(data_size=1_000_000, method="cubic")
print(f"Selected backend: {BackendManager.get_backend().name}")

# Create interpolator and keyframes
interpolator = KeyframeInterpolator()  # Default normalized 0-1 range
interpolator.set_keyframe(0.0, 0)
interpolator.set_keyframe(1.0, 10)

# Generate 1 million samples efficiently using the best available backend
samples = interpolator.sample_with_gpu(0, 1, 1_000_000, method="cubic")
```

### Exporting to Shader Code

Splinaltap can export your keyframe interpolation as shader code for use in graphics applications:

```python
from splinaltap import KeyframeInterpolator

# Create interpolator with keyframes
interpolator = KeyframeInterpolator()  # Normalized 0-1 range
interpolator.set_keyframe(0.0, 0)
interpolator.set_keyframe(0.25, 5)
interpolator.set_keyframe(0.75, 2)
interpolator.set_keyframe(1.0, 10)

# Export as GLSL shader function
glsl_code = interpolator.export_function(language="glsl", method="linear")
print(glsl_code)

# Export as C/C++ function
c_code = interpolator.export_function(language="c", method="linear")
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
| Multiple properties | < 1,000 properties | > 1,000 properties |
| Interactive editing | Real-time adjustments | Batch processing |
| Small animations | Few frames/keyframes | Complex render pipelines |

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
from splinaltap import KeyframeInterpolator
from splinaltap.backends import BackendManager

# Let splinaltap choose the best backend for your system
BackendManager.use_best_available()

# Create a complex interpolation
interpolator = KeyframeInterpolator()  # Normalized 0-1 range
interpolator.set_keyframe(0.0, 0)
interpolator.set_keyframe(0.25, "sin(t) + 1")
interpolator.set_keyframe(0.5, "t^2")
interpolator.set_keyframe(1.0, 10)

# Generate samples with optimal performance
samples = interpolator.sample_range(0, 1, 100_000, method="cubic")
```

## Applications

- Animation systems
- Scientific data interpolation
- Audio parameter automation
- Financial data modeling
- Simulation systems
- Game development
- Procedural content generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
