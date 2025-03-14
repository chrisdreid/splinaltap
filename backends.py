"""
Math backend support for splinaltap.

This module provides a uniform interface to different math backends:
- Pure Python (always available)
- NumPy (for CPU acceleration)
- CuPy (for GPU acceleration)
- JAX (for GPU acceleration and auto-differentiation)
- Numba (for JIT compilation on CPU and GPU)

Each backend has different trade-offs in terms of performance, features, and 
hardware requirements. The BackendManager allows the user to select the best
backend for their needs.
"""

import math
import warnings
import time
from typing import Dict, List, Optional, Union, Any, Type, Callable, Tuple

# Backend availability flags
HAS_NUMPY = False
HAS_CUPY = False
HAS_JAX = False
HAS_NUMBA = False
HAS_NUMBA_CUDA = False

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    pass

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    pass

# Try to import Numba
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    
    # Check for CUDA support in Numba
    try:
        from numba import cuda
        test_cuda = cuda.is_available()
        if test_cuda:
            HAS_NUMBA_CUDA = True
    except (ImportError, Exception):
        pass
except ImportError:
    pass


class BackendError(Exception):
    """Exception raised for backend-related errors."""
    pass


class Backend:
    """Base class for math backends."""
    
    name = "base"
    is_available = False
    supports_gpu = False
    supports_autodiff = False
    
    # Math functions
    sin = None
    cos = None
    tan = None
    exp = None
    log = None
    sqrt = None
    pow = None
    
    # Constants
    pi = None
    e = None
    
    # Array operations
    array = None
    zeros = None
    ones = None
    linspace = None
    arange = None
    
    # Linear algebra
    dot = None
    solve = None
    
    @classmethod
    def setup(cls) -> None:
        """Set up the backend (if needed)."""
        pass
    
    @classmethod
    def to_native_array(cls, arr: Any) -> Any:
        """Convert an array to the backend's native format."""
        raise NotImplementedError
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert a backend array to NumPy."""
        raise NotImplementedError
        
    @classmethod
    def performance_rank(cls, data_size: int = 1000, method: str = "linear") -> int:
        """Get the relative performance rank of this backend for a specific workload.
        
        Args:
            data_size: Estimated number of data points
            method: Interpolation method (determines complexity)
            
        Returns:
            Performance rank (higher is better)
        """
        if not cls.is_available:
            return 0
            
        # Base ranks:
        # Python: 1
        # NumPy: 10
        # Numba CPU: 50
        # CuPy/JAX small data: 20
        # CuPy/JAX large data: 100
        
        base_rank = 1  # Python
        
        # Adjust for backend type
        if cls.name == "python":
            base_rank = 1
        elif cls.name == "numpy":
            base_rank = 10
        elif cls.name == "numba":
            base_rank = 50
        elif cls.name in ["cupy", "jax"]:
            # GPU backends have overhead - adjust ranking based on data size
            if data_size < 10000:
                base_rank = 20
            else:
                base_rank = 100
                
        # Adjust for interpolation method complexity
        method_complexity = {
            "nearest": 1.0,
            "linear": 1.0,
            "polynomial": 2.0,
            "quadratic": 1.5,
            "cubic": 1.5,
            "hermite": 1.8,
            "bezier": 1.8,
            "pchip": 1.8,
            "gaussian": 3.0
        }
        
        complexity_factor = method_complexity.get(method, 1.0)
        
        # More complex methods benefit more from GPU acceleration
        if cls.supports_gpu and complexity_factor > 1.0:
            base_rank *= 1.5
            
        # Penalize GPU for very small datasets (overhead dominates)
        if cls.supports_gpu and data_size < 1000:
            base_rank *= 0.5
            
        return int(base_rank)


class PythonBackend(Backend):
    """Pure Python math backend using the standard library."""
    
    name = "python"
    is_available = True
    supports_gpu = False
    supports_autodiff = False
    
    # Math functions
    sin = math.sin
    cos = math.cos
    tan = math.tan
    exp = math.exp
    log = math.log
    sqrt = math.sqrt
    pow = math.pow
    
    # Constants
    pi = math.pi
    e = math.e
    
    @classmethod
    def array(cls, data: List) -> List:
        """Create a Python list from data."""
        return list(data)
    
    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> List:
        """Create a list of zeros."""
        if isinstance(shape, int):
            return [0.0] * shape
        
        # For multidimensional arrays, we need to use nested lists
        result = []
        if len(shape) == 1:
            return [0.0] * shape[0]
        else:
            return [cls.zeros(shape[1:]) for _ in range(shape[0])]
    
    @classmethod
    def ones(cls, shape: Union[int, Tuple[int, ...]]) -> List:
        """Create a list of ones."""
        if isinstance(shape, int):
            return [1.0] * shape
        
        # For multidimensional arrays, we need to use nested lists
        result = []
        if len(shape) == 1:
            return [1.0] * shape[0]
        else:
            return [cls.ones(shape[1:]) for _ in range(shape[0])]
    
    @classmethod
    def linspace(cls, start: float, stop: float, num: int) -> List[float]:
        """Create a list of evenly spaced numbers over an interval."""
        if num < 2:
            return [start]
        
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    
    @classmethod
    def arange(cls, start: float, stop: float, step: float = 1.0) -> List[float]:
        """Create a list of evenly spaced numbers within a range."""
        result = []
        current = start
        while current < stop:
            result.append(current)
            current += step
        return result
    
    @classmethod
    def dot(cls, a: List, b: List) -> float:
        """Compute the dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))
    
    @classmethod
    def solve(cls, a: List[List[float]], b: List[float]) -> List[float]:
        """Solve a linear system Ax = b for x. 
        
        This is a very basic implementation using Gaussian elimination.
        For anything complex, NumPy should be used instead.
        """
        n = len(a)
        
        # Create augmented matrix [A|b]
        aug = [row[:] + [b[i]] for i, row in enumerate(a)]
        
        # Gaussian elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(aug[max_row][i]):
                    max_row = j
            
            # Swap rows
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Eliminate below
            for j in range(i + 1, n):
                factor = aug[j][i] / aug[i][i]
                for k in range(i, n + 1):
                    aug[j][k] -= factor * aug[i][k]
        
        # Back substitution
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]
        
        return x
    
    @classmethod
    def to_native_array(cls, arr: Any) -> List:
        """Convert an array to a Python list."""
        if isinstance(arr, list):
            return arr
        
        # Handle numpy arrays
        if HAS_NUMPY and isinstance(arr, np.ndarray):
            return arr.tolist()
        
        # Handle other array-like objects
        return list(arr)
    
    @classmethod
    def to_numpy(cls, arr: List) -> Any:
        """Convert a Python list to NumPy if available."""
        if HAS_NUMPY:
            return np.array(arr)
        else:
            raise BackendError("NumPy is not available")


class NumpyBackend(Backend):
    """NumPy math backend for CPU acceleration."""
    
    name = "numpy"
    is_available = HAS_NUMPY
    supports_gpu = False
    supports_autodiff = False
    
    # Math functions
    sin = np.sin if HAS_NUMPY else None
    cos = np.cos if HAS_NUMPY else None
    tan = np.tan if HAS_NUMPY else None
    exp = np.exp if HAS_NUMPY else None
    log = np.log if HAS_NUMPY else None
    sqrt = np.sqrt if HAS_NUMPY else None
    pow = np.power if HAS_NUMPY else None
    
    # Constants
    pi = np.pi if HAS_NUMPY else None
    e = np.e if HAS_NUMPY else None
    
    # Array operations
    array = np.array if HAS_NUMPY else None
    zeros = np.zeros if HAS_NUMPY else None
    ones = np.ones if HAS_NUMPY else None
    linspace = np.linspace if HAS_NUMPY else None
    arange = np.arange if HAS_NUMPY else None
    
    # Linear algebra
    dot = np.dot if HAS_NUMPY else None
    solve = np.linalg.solve if HAS_NUMPY else None
    
    @classmethod
    def to_native_array(cls, arr: Any) -> Any:
        """Convert an array to a NumPy array."""
        if not HAS_NUMPY:
            raise BackendError("NumPy is not available")
            
        if isinstance(arr, np.ndarray):
            return arr
            
        return np.array(arr)
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert a NumPy array to NumPy (identity operation)."""
        if not HAS_NUMPY:
            raise BackendError("NumPy is not available")
            
        return arr


class CupyBackend(Backend):
    """CuPy math backend for GPU acceleration."""
    
    name = "cupy"
    is_available = HAS_CUPY
    supports_gpu = True
    supports_autodiff = False
    
    # Math functions
    sin = cp.sin if HAS_CUPY else None
    cos = cp.cos if HAS_CUPY else None
    tan = cp.tan if HAS_CUPY else None
    exp = cp.exp if HAS_CUPY else None
    log = cp.log if HAS_CUPY else None
    sqrt = cp.sqrt if HAS_CUPY else None
    pow = cp.power if HAS_CUPY else None
    
    # Constants
    pi = cp.pi if HAS_CUPY else None
    e = cp.e if HAS_CUPY else None
    
    # Array operations
    array = cp.array if HAS_CUPY else None
    zeros = cp.zeros if HAS_CUPY else None
    ones = cp.ones if HAS_CUPY else None
    linspace = cp.linspace if HAS_CUPY else None
    arange = cp.arange if HAS_CUPY else None
    
    # Linear algebra
    dot = cp.dot if HAS_CUPY else None
    solve = cp.linalg.solve if HAS_CUPY else None
    
    @classmethod
    def setup(cls) -> None:
        """Set up the CuPy backend."""
        if not HAS_CUPY:
            raise BackendError("CuPy is not available")
        
        # You could add device selection logic here if needed
        pass
    
    @classmethod
    def to_native_array(cls, arr: Any) -> Any:
        """Convert an array to a CuPy array."""
        if not HAS_CUPY:
            raise BackendError("CuPy is not available")
            
        if isinstance(arr, cp.ndarray):
            return arr
            
        # Convert from NumPy if needed
        if HAS_NUMPY and isinstance(arr, np.ndarray):
            return cp.array(arr)
            
        return cp.array(arr)
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert a CuPy array to NumPy."""
        if not HAS_CUPY:
            raise BackendError("CuPy is not available")
            
        if not HAS_NUMPY:
            raise BackendError("NumPy is not available for conversion from CuPy")
        
        # Check if it's a CuPy array 
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        
        # If it's already a list or other basic type, just return it 
        if isinstance(arr, (list, tuple, int, float)):
            return arr
            
        # For anything else, try generic conversion
        try:
            return cp.asnumpy(arr)
        except:
            # If conversion fails, return the original array
            return arr


class JaxBackend(Backend):
    """JAX math backend for GPU acceleration and autodifferentiation."""
    
    name = "jax"
    is_available = HAS_JAX
    supports_gpu = True
    supports_autodiff = True
    
    # Math functions
    sin = jnp.sin if HAS_JAX else None
    cos = jnp.cos if HAS_JAX else None
    tan = jnp.tan if HAS_JAX else None
    exp = jnp.exp if HAS_JAX else None
    log = jnp.log if HAS_JAX else None
    sqrt = jnp.sqrt if HAS_JAX else None
    pow = jnp.power if HAS_JAX else None
    
    # Constants
    pi = jnp.pi if HAS_JAX else None
    e = jnp.e if HAS_JAX else None
    
    # Array operations
    array = jnp.array if HAS_JAX else None
    zeros = jnp.zeros if HAS_JAX else None
    ones = jnp.ones if HAS_JAX else None
    linspace = jnp.linspace if HAS_JAX else None
    arange = jnp.arange if HAS_JAX else None
    
    # Linear algebra
    dot = jnp.dot if HAS_JAX else None
    solve = jnp.linalg.solve if HAS_JAX else None
    
    @classmethod
    def get_math_functions(cls):
        """Get common math functions from this backend.
        
        Returns:
            Dictionary of math functions and constants.
        """
        import random
        
        # Define the randint function to handle both forms:
        # randint([min, max]) - returns random int between min and max (inclusive)
        # randint(max) - returns random int between 0 and max (inclusive)
        def randint_func(min_max):
            if isinstance(min_max, (list, tuple)) and len(min_max) >= 2:
                return random.randint(int(min_max[0]), int(min_max[1]))
            else:
                return random.randint(0, int(min_max))
        
        # Handle JAX/NumPy array conversion to native types
        def safe_result_func(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                # If result is a JAX/NumPy array with a single value, convert to Python scalar
                if hasattr(result, 'item') and hasattr(result, 'size') and result.size == 1:
                    return result.item()
                return result
            return wrapper
        
        # Wrap all backend functions to ensure they return Python scalar values
        math_functions = {
            'sin': safe_result_func(cls.sin),
            'cos': safe_result_func(cls.cos),
            'tan': safe_result_func(cls.tan),
            'sqrt': safe_result_func(cls.sqrt), 
            'log': safe_result_func(cls.log), 
            'exp': safe_result_func(cls.exp),
            'pow': safe_result_func(cls.pow),
            'pi': cls.pi if not hasattr(cls.pi, 'item') else cls.pi.item(),
            'e': cls.e if not hasattr(cls.e, 'item') else cls.e.item(),
            'rand': random.random,
            'randint': randint_func
        }
        
        return math_functions
    
    @classmethod
    def setup(cls) -> None:
        """Set up the JAX backend."""
        if not HAS_JAX:
            raise BackendError("JAX is not available")
        
        # Set JAX config if needed
        pass
    
    @classmethod
    def to_native_array(cls, arr: Any) -> Any:
        """Convert an array to a JAX array."""
        if not HAS_JAX:
            raise BackendError("JAX is not available")
            
        if isinstance(arr, jnp.ndarray):
            return arr
            
        return jnp.array(arr)
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert a JAX array to NumPy."""
        if not HAS_JAX:
            raise BackendError("JAX is not available")
            
        if not HAS_NUMPY:
            raise BackendError("NumPy is not available for conversion from JAX")
        
        # Check if it's a JAX array 
        if isinstance(arr, jnp.ndarray):
            return np.array(arr)
        
        # If it's already a list or other basic type, just return it 
        if isinstance(arr, (list, tuple, int, float)):
            return arr
            
        # For anything else, try generic conversion
        try:
            return np.array(arr)
        except:
            # If conversion fails, return the original array
            return arr


class NumbaBackend(Backend):
    """Numba JIT-compiled math backend for CPU and GPU acceleration."""
    
    name = "numba"
    is_available = HAS_NUMBA
    supports_gpu = HAS_NUMBA_CUDA
    supports_autodiff = False
    
    # Numba requires NumPy
    if not HAS_NUMPY:
        is_available = False
    
    # Math functions - use NumPy's versions since Numba works with them
    sin = np.sin if HAS_NUMPY else None
    cos = np.cos if HAS_NUMPY else None
    tan = np.tan if HAS_NUMPY else None
    exp = np.exp if HAS_NUMPY else None
    log = np.log if HAS_NUMPY else None
    sqrt = np.sqrt if HAS_NUMPY else None
    pow = np.power if HAS_NUMPY else None
    
    # Constants
    pi = np.pi if HAS_NUMPY else None
    e = np.e if HAS_NUMPY else None
    
    # Array operations - use NumPy as base
    array = np.array if HAS_NUMPY else None
    zeros = np.zeros if HAS_NUMPY else None
    ones = np.ones if HAS_NUMPY else None
    linspace = np.linspace if HAS_NUMPY else None
    arange = np.arange if HAS_NUMPY else None
    
    # Linear algebra
    dot = np.dot if HAS_NUMPY else None
    solve = np.linalg.solve if HAS_NUMPY else None
    
    # JIT-compiled functions
    _jit_functions = {}
    
    @classmethod
    def setup(cls) -> None:
        """Set up the Numba backend."""
        if not HAS_NUMBA:
            raise BackendError("Numba is not available")
        if not HAS_NUMPY:
            raise BackendError("NumPy is required for Numba backend")
        
        # Initialize JIT functions if needed
        pass
    
    @classmethod
    def jit(cls, func):
        """JIT-compile a function using Numba."""
        if not HAS_NUMBA:
            return func
        
        # Cache the function if we've already compiled it
        if func in cls._jit_functions:
            return cls._jit_functions[func]
        
        # Compile with Numba and cache it
        if HAS_NUMBA_CUDA and func.__name__.endswith('_gpu'):
            # CUDA implementation
            jitted = cuda.jit(func)
        else:
            # CPU implementation
            jitted = jit(nopython=True, parallel=True)(func)
        
        cls._jit_functions[func] = jitted
        return jitted
    
    @classmethod
    def to_native_array(cls, arr: Any) -> Any:
        """Convert an array to a NumPy array for Numba."""
        if not HAS_NUMBA:
            raise BackendError("Numba is not available")
        
        if isinstance(arr, np.ndarray):
            return arr
            
        return np.array(arr)
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert an array to NumPy (identity for Numba)."""
        if not HAS_NUMBA:
            raise BackendError("Numba is not available")
            
        return arr  # Already NumPy arrays


class BackendManager:
    """Manager for selecting and using math backends."""
    
    _backends = {
        "python": PythonBackend,
        "numpy": NumpyBackend,
        "cupy": CupyBackend,
        "jax": JaxBackend,
        "numba": NumbaBackend
    }
    
    _current_backend = PythonBackend
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """Get a list of available backends."""
        return [name for name, backend in cls._backends.items() if backend.is_available]
    
    @classmethod
    def get_backend(cls, name: Optional[str] = None) -> Type[Backend]:
        """Get a backend by name, or the current backend if name is None."""
        if name is None:
            return cls._current_backend
            
        if name not in cls._backends:
            raise BackendError(f"Unknown backend: {name}")
            
        backend = cls._backends[name]
        if not backend.is_available:
            raise BackendError(f"Backend {name} is not available")
            
        return backend
    
    @classmethod
    def set_backend(cls, name: str) -> None:
        """Set the current backend."""
        backend = cls.get_backend(name)
        try:
            backend.setup()
            cls._current_backend = backend
        except Exception as e:
            raise BackendError(f"Failed to set backend {name}: {e}")
    
    @classmethod
    def get_best_available_backend(cls, data_size: int = 1000, method: str = "linear") -> Type[Backend]:
        """Get the best available backend based on system capabilities and workload.
        
        Args:
            data_size: Estimated number of data points to process
            method: Interpolation method to use (affects complexity)
            
        Returns:
            The best backend class for the given workload
        """
        available = cls.available_backends()
        if not available:
            return PythonBackend
            
        # Rank available backends for this specific workload
        ranked_backends = []
        for name in available:
            backend = cls._backends[name]
            try:
                rank = backend.performance_rank(data_size, method)
                ranked_backends.append((rank, name))  # Store name instead of class
            except Exception as e:
                print(f"Warning: Error ranking backend {name}: {e}")
                # Still include with a low rank
                ranked_backends.append((0, name))
            
        # Sort by rank (descending)
        ranked_backends.sort(reverse=True, key=lambda x: x[0])  # Sort only by rank
        
        # Return the highest-ranked backend
        best_name = ranked_backends[0][1]
        return cls._backends[best_name]
    
    @classmethod
    def use_best_available(cls, data_size: int = 1000, method: str = "linear") -> None:
        """Switch to the best available backend for the given workload.
        
        Args:
            data_size: Estimated number of data points to process
            method: Interpolation method to use (affects complexity)
        """
        backend = cls.get_best_available_backend(data_size, method)
        cls.set_backend(backend.name)
        
    @classmethod
    def benchmark_backends(cls, data_size: int = 10000, method: str = "linear", repetitions: int = 3) -> Dict[str, float]:
        """Benchmark all available backends for a specific workload.
        
        Args:
            data_size: Number of data points to test with
            method: Interpolation method to test
            repetitions: Number of test repetitions for accuracy
            
        Returns:
            Dictionary mapping backend names to execution times (seconds)
        """
        # Create test data - simple linear interpolation between 0 and 1
        x_values = [0.0, 1.0]
        y_values = [0.0, 1.0]
        sample_points = [i / (data_size - 1) for i in range(data_size)]
        
        results = {}
        original_backend = cls.get_backend().name
        
        for backend_name in cls.available_backends():
            try:
                # Switch to this backend
                cls.set_backend(backend_name)
                backend = cls.get_backend()
                
                # Run warmup iteration
                cls._benchmark_iteration(backend, x_values, y_values, sample_points, method)
                
                # Run timed iterations
                times = []
                for _ in range(repetitions):
                    elapsed = cls._benchmark_iteration(backend, x_values, y_values, sample_points, method)
                    times.append(elapsed)
                
                # Record average time
                results[backend_name] = sum(times) / len(times)
                
            except Exception as e:
                results[backend_name] = float('inf')  # Mark as failed
                warnings.warn(f"Benchmark failed for {backend_name}: {e}")
                
        # Restore original backend
        cls.set_backend(original_backend)
        return results
    
    @classmethod
    def _benchmark_iteration(cls, backend: Type[Backend], x_values, y_values, sample_points, method: str) -> float:
        """Run a single benchmark iteration for a backend."""
        # Prepare arrays in this backend's format
        x_arr = backend.array(x_values)
        y_arr = backend.array(y_values)
        sample_arr = backend.array(sample_points)
        result_arr = backend.zeros(len(sample_points))
        
        # Time basic linear interpolation
        start_time = time.time()
        
        # Simplified linear interpolation for benchmarking
        for i in range(len(sample_points)):
            t = sample_points[i]
            # Simple linear interpolation: y = y0 + (y1-y0)*(t-x0)/(x1-x0)
            result_arr[i] = y_arr[0] + (y_arr[1] - y_arr[0]) * (t - x_arr[0]) / (x_arr[1] - x_arr[0])
            
        end_time = time.time()
        return end_time - start_time
    
    # Forward commonly used functions to the current backend
    @classmethod
    def array(cls, data: Any) -> Any:
        """Create an array with the current backend."""
        return cls._current_backend.array(data)
    
    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> Any:
        """Create an array of zeros with the current backend."""
        return cls._current_backend.zeros(shape)
    
    @classmethod
    def ones(cls, shape: Union[int, Tuple[int, ...]]) -> Any:
        """Create an array of ones with the current backend."""
        return cls._current_backend.ones(shape)
    
    @classmethod
    def linspace(cls, start: float, stop: float, num: int) -> Any:
        """Create a linearly spaced array with the current backend."""
        return cls._current_backend.linspace(start, stop, num)
    
    @classmethod
    def to_numpy(cls, arr: Any) -> Any:
        """Convert an array to NumPy."""
        # Fast path for common types, avoid unnecessary conversions
        if isinstance(arr, (list, tuple, int, float)) or (HAS_NUMPY and isinstance(arr, np.ndarray)):
            return arr
            
        try:
            # Let the backend handle the conversion
            result = cls._current_backend.to_numpy(arr)
            return result
        except Exception as e:
            # If any conversion error occurs, just return the original array
            warnings.warn(f"Array conversion failed: {e}, returning original array")
            return arr


# Initialize with the best available backend
BackendManager.use_best_available()


def get_math_functions():
    """Get common math functions from the current backend.
    
    Returns:
        Dictionary of math functions and constants from the current backend.
    """
    import random
    
    backend = BackendManager.get_backend()
    
    # Define the randint function to handle both forms:
    # randint([min, max]) - returns random int between min and max (inclusive)
    # randint(max) - returns random int between 0 and max (inclusive)
    def randint_func(min_max):
        if isinstance(min_max, (list, tuple)) and len(min_max) >= 2:
            return random.randint(int(min_max[0]), int(min_max[1]))
        else:
            return random.randint(0, int(min_max))
    
    # Handle NumPy array conversion to native types
    def safe_result_func(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # If result is a NumPy array with a single value, convert to Python scalar
            if hasattr(result, 'item') and hasattr(result, 'size') and result.size == 1:
                return result.item()
            return result
        return wrapper
    
    # Wrap all backend functions to ensure they return Python scalar values
    math_functions = {
        'sin': safe_result_func(backend.sin),
        'cos': safe_result_func(backend.cos),
        'tan': safe_result_func(backend.tan),
        'sqrt': safe_result_func(backend.sqrt), 
        'log': safe_result_func(backend.log), 
        'exp': safe_result_func(backend.exp),
        'pow': safe_result_func(backend.pow),
        'pi': backend.pi if not hasattr(backend.pi, 'item') else backend.pi.item(),
        'e': backend.e if not hasattr(backend.e, 'item') else backend.e.item(),
        'rand': random.random,
        'randint': randint_func
    }
    
    return math_functions