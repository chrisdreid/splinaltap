import ast
from typing import Dict, Union, List, Callable, Tuple, Optional, Sequence, Any

from .backends import BackendManager, get_math_functions

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
        # Get math functions from the current backend
        math_funcs = get_math_functions()
        
        safe_dict = {
            'sin': math_funcs['sin'], 
            'cos': math_funcs['cos'], 
            'tan': math_funcs['tan'],
            'sqrt': math_funcs['sqrt'], 
            'log': math_funcs['log'], 
            'exp': math_funcs['exp'],
            'pi': math_funcs['pi'], 
            'e': math_funcs['e'], 
            'pow': math_funcs['pow'], 
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
                    method: str = "linear", channels: Dict[str, float] = {}) -> Union[List[float], Any]:
        """Sample values at evenly spaced intervals within a time range.
        
        Args:
            start_time: Start time to sample from
            end_time: End time to sample to
            num_samples: Number of samples to generate
            method: Interpolation method to use
            channels: Dictionary of channel values to use in expressions
            
        Returns:
            Array of sampled values (using the current backend)
        """
        if num_samples < 2:
            raise ValueError("Number of samples must be at least 2")
        
        # Create time values array using current backend
        t_values = BackendManager.linspace(start_time, end_time, num_samples)
        
        # Create output array using current backend
        output = BackendManager.zeros(num_samples)
        
        # Sample values
        self.sample_to_array(output, start_time, end_time, method, channels)
        
        return output
        
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
        num_samples = len(output_array)
        step = (end_time - start_time) / (num_samples - 1) if num_samples > 1 else 0
        
        # Simple implementation that works with any array type
        for i in range(num_samples):
            t = start_time + i * step
            output_array[i] = self.get_value(t, method, channels)
            
    def sample_with_gpu(self, start_time: float, end_time: float, num_samples: int,
                      method: str = "linear", channels: Dict[str, float] = {}) -> Any:
        """Sample values using GPU acceleration if available.
        
        This method attempts to use CuPy, JAX, or Numba for accelerated sampling.
        If none are available, falls back to regular CPU sampling.
        
        Args:
            start_time: Start time to sample from
            end_time: End time to sample to
            num_samples: Number of samples to generate
            method: Interpolation method to use
            channels: Dictionary of channel values to use in expressions
            
        Returns:
            Array of sampled values
        """
        # Select the best backend for this workload
        original_backend = BackendManager.get_backend().name
        
        # Use the performance ranking system to choose the best backend
        BackendManager.use_best_available(data_size=num_samples, method=method)
        current_backend = BackendManager.get_backend()
        result = None
        
        try:
            # Get keyframe points as arrays
            points = self._get_keyframe_points(channels)
            times = BackendManager.array([p[0] for p in points])
            values = BackendManager.array([p[1] for p in points])
            
            # Get derivatives for methods that need them (hermite, etc.)
            derivatives = None
            if method in ["hermite", "pchip"]:
                derivatives = BackendManager.array([
                    self.keyframes[index][1] if self.keyframes[index][1] is not None else 0.0 
                    for index in sorted(self.keyframes)
                ])
            
            # Get control points for Bezier
            control_points = None
            if method == "bezier":
                # Collect all control points
                ctrl_pts = []
                for index in sorted(self.keyframes):
                    if self.keyframes[index][2] is not None:
                        ctrl_pts.extend(self.keyframes[index][2])
                    else:
                        # Default control points if not specified
                        val = self._evaluate_keyframe(index, index, channels)
                        ctrl_pts.extend([index, val, index, val])
                
                if ctrl_pts:
                    control_points = BackendManager.array(ctrl_pts)
            
            # Create sample times
            t_array = BackendManager.linspace(start_time, end_time, num_samples)
            result = BackendManager.zeros(num_samples)
            
            # Apply the appropriate interpolation method
            if method == "nearest":
                result = self._nearest_interpolate_gpu(times, values, t_array)
            elif method == "linear":
                result = self._linear_interpolate_gpu(times, values, t_array)
            elif method == "polynomial":
                result = self._polynomial_interpolate_gpu(times, values, t_array)
            elif method == "quadratic":
                result = self._quadratic_interpolate_gpu(times, values, t_array)
            elif method == "cubic":
                result = self._cubic_interpolate_gpu(times, values, t_array)
            elif method == "hermite":
                result = self._hermite_interpolate_gpu(times, values, derivatives, t_array)
            elif method == "bezier":
                result = self._bezier_interpolate_gpu(times, values, control_points, t_array)
            elif method == "pchip":
                result = self._pchip_interpolate_gpu(times, values, derivatives, t_array)
            elif method == "gaussian":
                result = self._gaussian_interpolate_gpu(times, values, t_array)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
                
        finally:
            # Restore the original backend
            BackendManager.set_backend(original_backend)
            
        if result is not None:
            # Convert to the current backend's format
            return BackendManager.get_backend().to_native_array(result)
        
        # Fall back to regular sampling if acceleration failed
        return self.sample_range(start_time, end_time, num_samples, method, channels)
        
    def _nearest_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated nearest neighbor interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For each sample point, find the nearest keyframe
        if backend.name in ["cupy", "jax"]:
            # Vectorized implementation for CuPy/JAX
            for i in range(len(t_array)):
                t = t_array[i]
                # Find closest time point
                abs_diff = backend.abs(times - t)
                closest_idx = backend.argmin(abs_diff)
                result[i] = values[closest_idx]
        else:
            # Loop implementation for other backends
            for i in range(len(t_array)):
                t = t_array[i]
                closest_idx = 0
                min_diff = float('inf')
                for j in range(len(times)):
                    diff = abs(times[j] - t)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = j
                result[i] = values[closest_idx]
                
        return result
    
    def _linear_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated linear interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For each sample point, find the bracketing keyframes and interpolate
        for i in range(len(t_array)):
            t = t_array[i]
            if t <= times[0]:
                result[i] = values[0]
            elif t >= times[-1]:
                result[i] = values[-1]
            else:
                # Find the indices of the keyframes that bracket this time
                for j in range(len(times) - 1):
                    if times[j] <= t <= times[j + 1]:
                        # Linear interpolation
                        alpha = (t - times[j]) / (times[j + 1] - times[j])
                        result[i] = values[j] * (1 - alpha) + values[j + 1] * alpha
                        break
                        
        return result
    
    def _polynomial_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated polynomial (Lagrange) interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        n = len(times)
        
        # For each sample point, compute the Lagrange polynomial
        for i in range(len(t_array)):
            t = t_array[i]
            if t <= times[0]:
                result[i] = values[0]
            elif t >= times[-1]:
                result[i] = values[-1]
            else:
                # Lagrange interpolation
                sum_value = 0.0
                for j in range(n):
                    # Compute the Lagrange basis polynomial
                    basis = 1.0
                    for k in range(n):
                        if k != j:
                            basis *= (t - times[k]) / (times[j] - times[k])
                    sum_value += values[j] * basis
                result[i] = sum_value
                
        return result
    
    def _quadratic_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated quadratic spline interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        n = len(times)
        
        # Need at least 3 points for quadratic interpolation
        if n < 3:
            return self._linear_interpolate_gpu(times, values, t_array)
            
        # Simplified implementation - just finds closest 3 points for interpolation
        for i in range(len(t_array)):
            t = t_array[i]
            if t <= times[0]:
                result[i] = values[0]
            elif t >= times[-1]:
                result[i] = values[-1]
            else:
                # Find the segment this t is in
                segment = 0
                for j in range(n - 1):
                    if times[j] <= t <= times[j + 1]:
                        segment = j
                        break
                
                # Get 3 points for quadratic interpolation
                idx1 = max(0, segment - 1)
                idx2 = segment
                idx3 = min(n - 1, segment + 1)
                
                # Simple quadratic through 3 points
                x1, y1 = times[idx1], values[idx1]
                x2, y2 = times[idx2], values[idx2]
                x3, y3 = times[idx3], values[idx3]
                
                # Quadratic interpolation coefficients
                denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
                a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
                b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
                c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
                
                # Evaluate quadratic at t
                result[i] = a * t*t + b * t + c
                
        return result
    
    # Add implementations for other interpolation methods (_cubic_interpolate_gpu, _hermite_interpolate_gpu, etc.)
    # These would follow a similar pattern but with the specific math for each method
    
    def _cubic_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated cubic spline interpolation."""
        # Simplified implementation - for a proper cubic spline, we'd need to solve a tridiagonal system
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For now, this is a simpler cubic interpolation (not a true spline)
        for i in range(len(t_array)):
            t = t_array[i]
            # Use regular method for now - would be replaced with proper GPU implementation
            result[i] = cubic_spline(self, t, {})
            
        return result
    
    def _hermite_interpolate_gpu(self, times, values, derivatives, t_array):
        """GPU-accelerated hermite interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For now, just use the regular method - would be replaced with proper GPU implementation
        for i in range(len(t_array)):
            t = t_array[i]
            result[i] = hermite_interpolate(self, t, {})
            
        return result
        
    def _bezier_interpolate_gpu(self, times, values, control_points, t_array):
        """GPU-accelerated bezier interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For now, just use the regular method - would be replaced with proper GPU implementation
        for i in range(len(t_array)):
            t = t_array[i]
            result[i] = bezier_interpolate(self, t, {})
            
        return result
    
    def _pchip_interpolate_gpu(self, times, values, derivatives, t_array):
        """GPU-accelerated PCHIP interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For now, just use the regular method - would be replaced with proper GPU implementation
        for i in range(len(t_array)):
            t = t_array[i]
            result[i] = pchip_interpolate(self, t, {})
            
        return result
    
    def _gaussian_interpolate_gpu(self, times, values, t_array):
        """GPU-accelerated gaussian process interpolation."""
        backend = BackendManager.get_backend()
        result = backend.zeros(len(t_array))
        
        # For now, just use the regular method - would be replaced with proper GPU implementation
        for i in range(len(t_array)):
            t = t_array[i]
            result[i] = gaussian_interpolate(self, t, {})
            
        return result
                
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range covered by keyframes.
        
        Returns:
            Tuple of (min_time, max_time)
        """
        if not self.keyframes:
            raise ValueError("No keyframes defined")
            
        times = sorted(self.keyframes.keys())
        return (times[0], times[-1])
        
    def export_function(self, language: str = "glsl", method: str = "linear") -> str:
        """Export interpolation as a function in the specified language.
        
        This allows the interpolation to be used in shaders or other environments
        where Python code cannot run directly.
        
        Args:
            language: The target language ("glsl", "hlsl", "wgsl", "cuda", "c")
            method: Interpolation method to use
            
        Returns:
            String containing the interpolation function in the target language
        """
        if not self.keyframes:
            raise ValueError("No keyframes defined")
            
        # Get keyframe points
        points = self._get_keyframe_points()
        times = [p[0] for p in points]
        values = [p[1] for p in points]
        
        # Get function name based on method
        func_name = f"{method}Interpolate"
        
        if language == "glsl":
            return self._export_glsl(func_name, times, values, method)
        elif language == "hlsl":
            return self._export_hlsl(func_name, times, values, method)
        elif language == "wgsl":
            return self._export_wgsl(func_name, times, values, method)
        elif language == "cuda":
            return self._export_cuda(func_name, times, values, method)
        elif language == "c":
            return self._export_c(func_name, times, values, method)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _export_glsl(self, func_name: str, times: List[float], values: List[float], method: str) -> str:
        """Export interpolation as a GLSL function."""
        # For simple keyframes, we generate a straightforward GLSL function
        if method == "linear":
            # Create a GLSL function for linear interpolation
            code = [
                f"// GLSL linear interpolation function for {len(times)} keyframes",
                f"float {func_name}(float t) {{",
                "    // Keyframe times"
            ]
            
            # Write times array
            times_str = ", ".join([f"{t:.6f}" for t in times])
            code.append(f"    float times[{len(times)}] = float[{len(times)}]({times_str});")
            
            # Write values array
            values_str = ", ".join([f"{v:.6f}" for v in values])
            code.append(f"    float values[{len(values)}] = float[{len(values)}]({values_str});")
            
            # Simple linear interpolation logic
            code.extend([
                "",
                "    // Handle out-of-range times",
                f"    if (t <= times[0]) return values[0];",
                f"    if (t >= times[{len(times) - 1}]) return values[{len(times) - 1}];",
                "",
                "    // Find the bracketing keyframes",
                "    for (int i = 0; i < times.length() - 1; i++) {",
                "        if (times[i] <= t && t <= times[i + 1]) {",
                "            float alpha = (t - times[i]) / (times[i + 1] - times[i]);",
                "            return mix(values[i], values[i + 1], alpha);",
                "        }",
                "    }",
                "",
                "    // Fallback (should never reach here)",
                "    return values[0];",
                "}"
            ])
            
            return "\n".join(code)
            
        elif method == "cubic":
            # Create a GLSL function for cubic interpolation (simplified)
            # This would need to be expanded for real-world use
            code = [
                f"// GLSL cubic interpolation function for {len(times)} keyframes",
                f"float {func_name}(float t) {{",
                "    // Implementation of cubic interpolation would go here",
                "    // This is a simplified placeholder",
                "    return 0.0;",
                "}"
            ]
            return "\n".join(code)
            
        else:
            # Other methods would follow similar patterns
            return f"// GLSL export for {method} interpolation not yet implemented"
    
    def _export_hlsl(self, func_name: str, times: List[float], values: List[float], method: str) -> str:
        """Export interpolation as an HLSL function."""
        # HLSL implementation would be similar to GLSL
        return f"// HLSL export for {method} interpolation not yet implemented"
        
    def _export_wgsl(self, func_name: str, times: List[float], values: List[float], method: str) -> str:
        """Export interpolation as a WGSL function."""
        # WGSL implementation would be similar to GLSL with syntax adjustments
        return f"// WGSL export for {method} interpolation not yet implemented"
        
    def _export_cuda(self, func_name: str, times: List[float], values: List[float], method: str) -> str:
        """Export interpolation as a CUDA function."""
        # CUDA implementation would be more C-like
        return f"// CUDA export for {method} interpolation not yet implemented"
        
    def _export_c(self, func_name: str, times: List[float], values: List[float], method: str) -> str:
        """Export interpolation as a C function."""
        # Basic C implementation for linear interpolation
        if method == "linear":
            # Create a C function for linear interpolation
            code = [
                f"// C linear interpolation function for {len(times)} keyframes",
                f"float {func_name}(float t) {{",
                "    // Keyframe times and values"
            ]
            
            # Write times array
            times_arr = ", ".join([f"{t:.6f}f" for t in times])
            code.append(f"    static const float times[{len(times)}] = {{{times_arr}}};")
            
            # Write values array
            values_arr = ", ".join([f"{v:.6f}f" for v in values])
            code.append(f"    static const float values[{len(values)}] = {{{values_arr}}};")
            
            # Linear interpolation logic
            code.extend([
                f"    static const int count = {len(times)};",
                "",
                "    // Handle out-of-range times",
                "    if (t <= times[0]) return values[0];",
                "    if (t >= times[count - 1]) return values[count - 1];",
                "",
                "    // Find the bracketing keyframes",
                "    for (int i = 0; i < count - 1; i++) {",
                "        if (times[i] <= t && t <= times[i + 1]) {",
                "            float alpha = (t - times[i]) / (times[i + 1] - times[i]);",
                "            return values[i] * (1.0f - alpha) + values[i + 1] * alpha;",
                "        }",
                "    }",
                "",
                "    // Fallback (should never reach here)",
                "    return values[0];",
                "}"
            ])
            
            return "\n".join(code)
        else:
            # Other methods would be implemented similarly
            return f"// C export for {method} interpolation not yet implemented"