import math
import ast
from typing import Dict, Union, List, Callable, Tuple, Optional, Sequence, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .methods import (
    nearest_neighbor,
    linear_interpolate,
    polynomial_interpolate,
    quadratic_spline,
    hermite_interpolate,
    bezier_interpolate,
    gaussian_interpolate,
    pchip_interpolate,
    cubic_spline
)

class KeyframeInterpolator:
    def __init__(self, num_indices: Optional[int] = None, time_range: Optional[Tuple[float, float]] = None):
        """Initialize the keyframe interpolator.
        
        Args:
            num_indices: Optional total number of indices. If None, the interpolator
                         will work in continuous time mode without a fixed range.
            time_range: Optional tuple of (min_time, max_time) defining the time range.
                        If None, the range will be determined from the keyframes.
        """
        if num_indices is not None and num_indices < 1:
            raise ValueError("Number of indices must be at least 1")
        
        self.num_indices = float(num_indices) if num_indices is not None else None
        self.time_range = time_range
        self.keyframes: Dict[float, Tuple[Callable[[float, Dict[str, float]], float], Optional[float], Optional[Tuple[float, float, float, float]]]] = {}
        self.variables: Dict[str, Callable[[float, Dict[str, float]], float]] = {}
        self._precomputed = {}

    def _parse_expression(self, expr: str) -> Callable[[float, Dict[str, float]], float]:
        """Parse an expression into a safe lambda function."""
        safe_dict = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e, 'pow': math.pow, 
            'T': self.num_indices if self.num_indices is not None else 0
        }
        tree = ast.parse(expr, mode='eval')
        compiled = compile(tree, "<string>", "eval")

        class SafeVisitor(ast.NodeVisitor):
            def __init__(self, variables: Dict[str, Callable[[float, Dict[str, float]], float]]):
                self.variables = variables
                self.used_vars = set()

            allowed_nodes = {
                ast.Expression, ast.Num, ast.UnaryOp, ast.BinOp, ast.Name, ast.Call, ast.Load,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant,
                ast.IfExp, ast.Compare, ast.Eq, ast.Mod, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.NotEq,
                ast.BoolOp, ast.And
            }

            def generic_visit(self, node):
                if type(node) not in self.allowed_nodes:
                    raise ValueError(f"Unsafe operation: {type(node).__name__}")
                super().generic_visit(node)

            def visit_Name(self, node):
                if node.id in self.variables:
                    self.used_vars.add(node.id)
                elif node.id not in safe_dict and node.id != 't' and node.id not in 'abcd':
                    raise ValueError(f"Unknown name: {node.id}")
                super().generic_visit(node)

            def visit_Call(self, node):
                if not isinstance(node.func, ast.Name) or node.func.id not in safe_dict:
                    raise ValueError(f"Unsafe call: {node.func}")
                super().generic_visit(node)

        visitor = SafeVisitor(self.variables)
        visitor.visit(tree)
        used_vars = visitor.used_vars

        def lambda_func(t: float, channels: Dict[str, float] = {}) -> float:
            eval_dict = safe_dict | {'t': t} | channels
            for var in self.variables:
                if var in used_vars and var not in eval_dict:
                    eval_dict[var] = self.variables[var](t, channels)
            return eval(compiled, {"__builtins__": {}}, eval_dict)

        return lambda_func

    def set_keyframe(self, index: float, value: Union[int, float, str], derivative: Optional[float] = None, control_points: Optional[Tuple[float, float, float, float]] = None):
        """Set a keyframe with optional derivative and control points.
        
        Args:
            index: The time index for the keyframe (can be arbitrary time in ms)
            value: The value at this keyframe (can be a number or an expression)
            derivative: Optional derivative at this point (for Hermite interpolation)
            control_points: Optional tuple of control points (x1, y1, x2, y2) for Bezier interpolation
        """
        # In continuous time mode, we accept any time value
        if self.num_indices is not None and not 0 <= index <= self.num_indices:
            raise ValueError(f"Index {index} must be between 0 and {self.num_indices}")
            
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"Keyframe value must be an int, float, or string, got {type(value).__name__}")
            
        if isinstance(value, (int, float)):
            self.keyframes[index] = (lambda t, channels={}: float(value), derivative, control_points)
        else:
            self.keyframes[index] = (self._parse_expression(value), derivative, control_points)
            
        self._precomputed.clear()

    def set_variable(self, name: str, value: Union[int, float, str]):
        """Set a variable value (constant or expression) as a lambda function.
        
        Args:
            name: The variable name to be used in expressions
            value: The value or expression for this variable
        """
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"Variable value must be an int, float, or string, got {type(value).__name__}")
        if isinstance(value, (int, float)):
            self.variables[name] = lambda t, channels={}: float(value)
        else:
            self.variables[name] = self._parse_expression(value)

    def _evaluate_keyframe(self, index: float, t: float, channels: Dict[str, float] = {}) -> float:
        """Evaluate a keyframe at a given t with channel values."""
        return self.keyframes[index][0](t, channels)

    def _get_keyframe_points(self, channels: Dict[str, float] = {}) -> List[Tuple[float, float]]:
        """Convert keyframes to a list of (index, value) pairs with channel values."""
        return [(index, self._evaluate_keyframe(index, index, channels)) for index in sorted(self.keyframes)]

    def get_value(self, t: float, method: str = "linear", channels: Dict[str, float] = {}) -> float:
        """Get the interpolated value at time t using the specified method.
        
        Args:
            t: The time at which to evaluate
            method: Interpolation method to use (nearest, linear, polynomial, quadratic, 
                    hermite, bezier, gaussian, pchip, cubic)
            channels: Dictionary of channel values to use in expressions
            
        Returns:
            The interpolated value at time t
        """
        methods = {
            "nearest": nearest_neighbor,
            "linear": linear_interpolate,
            "polynomial": polynomial_interpolate,
            "quadratic": quadratic_spline,
            "hermite": hermite_interpolate,
            "bezier": bezier_interpolate,
            "gaussian": gaussian_interpolate,
            "pchip": pchip_interpolate,
            "cubic": cubic_spline
        }
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
        
        return methods[method](self, t, channels)
    
    def sample_range(self, start_time: float, end_time: float, num_samples: int, 
                    method: str = "linear", channels: Dict[str, float] = {}) -> List[float]:
        """Sample values at evenly spaced intervals within a time range.
        
        Args:
            start_time: Start time to sample from
            end_time: End time to sample to
            num_samples: Number of samples to generate
            method: Interpolation method to use
            channels: Dictionary of channel values to use in expressions
            
        Returns:
            List of sampled values
        """
        if num_samples < 2:
            raise ValueError("Number of samples must be at least 2")
            
        result = []
        step = (end_time - start_time) / (num_samples - 1) if num_samples > 1 else 0
        
        for i in range(num_samples):
            t = start_time + i * step
            result.append(self.get_value(t, method, channels))
            
        return result
        
    def sample_to_array(self, output_array: Union[List[float], Any], 
                      start_time: float, end_time: float, 
                      method: str = "linear", channels: Dict[str, float] = {}) -> None:
        """Sample values directly into an existing array.
        
        Args:
            output_array: Array to populate with sampled values
            start_time: Start time to sample from
            end_time: End time to sample to
            method: Interpolation method to use
            channels: Dictionary of channel values to use in expressions
        """
        if HAS_NUMPY and isinstance(output_array, np.ndarray):
            num_samples = len(output_array)
            step = (end_time - start_time) / (num_samples - 1) if num_samples > 1 else 0
            
            for i in range(num_samples):
                t = start_time + i * step
                output_array[i] = self.get_value(t, method, channels)
        else:
            # Handling standard Python lists
            num_samples = len(output_array)
            step = (end_time - start_time) / (num_samples - 1) if num_samples > 1 else 0
            
            for i in range(num_samples):
                t = start_time + i * step
                output_array[i] = self.get_value(t, method, channels)
                
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range covered by keyframes.
        
        Returns:
            Tuple of (min_time, max_time)
        """
        if not self.keyframes:
            raise ValueError("No keyframes defined")
            
        times = sorted(self.keyframes.keys())
        return (times[0], times[-1])