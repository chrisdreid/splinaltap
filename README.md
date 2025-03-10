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
- 🎛️ **Channel Support**: Pass in dynamic channel values that can be used in expressions
- 📊 **Visualization**: Built-in support for visualizing interpolation results
- 🔒 **Safe Execution**: No unsafe `eval()` - all expressions are parsed and evaluated securely

## Installation

```bash
pip install splinaltap
```

## Quick Start

```python
from splinaltap import KeyframeInterpolator
import matplotlib.pyplot as plt

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
values = [interpolator.get_value(t, "cubic") for t in t_values]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t_values, values)
plt.title("Cubic Spline Interpolation")
plt.grid(True)
plt.show()
```

## Advanced Usage

### Using Different Interpolation Methods

```python
# Compare different interpolation methods
methods = ["linear", "cubic", "hermite", "bezier"]
plt.figure(figsize=(12, 8))

for method in methods:
    values = [interpolator.get_value(t, method) for t in t_values]
    plt.plot(t_values, values, label=method.capitalize())

plt.legend()
plt.title("Interpolation Methods Comparison")
plt.show()
```

### Using Channels

```python
# Define keyframes that use channel values
interpolator.set_keyframe(3.0, "a * sin(t) + b")

# Evaluate with different channel values
channels_1 = {"a": 1.0, "b": 0.5}
channels_2 = {"a": 2.0, "b": 0.0}

values_1 = [interpolator.get_value(t, "cubic", channels_1) for t in t_values]
values_2 = [interpolator.get_value(t, "cubic", channels_2) for t in t_values]
```

### Using Control Points (Bezier)

```python
# Set keyframe with control points for Bezier interpolation
interpolator.set_keyframe(4.0, 5.0, derivative=None, control_points=(4.2, 6.0, 4.8, 7.0))
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
