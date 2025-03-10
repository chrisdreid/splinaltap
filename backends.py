"""
Math backend support for splinaltap.

This module provides a uniform interface to different math backends:
- Pure Python (always available)
- NumPy (for CPU acceleration)
- CuPy (for GPU acceleration)
- JAX (for GPU acceleration and auto-differentiation)

Each backend has different trade-offs in terms of performance, features, and 
hardware requirements. The BackendManager allows the user to select the best
backend for their needs.
"""

import math
import warnings
from typing import Dict, List, Optional, Union, Any, Type, Callable, Tuple

# Backend availability flags
HAS_NUMPY = False
HAS_CUPY = False
HAS_JAX = False

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
            
        return cp.asnumpy(arr)


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
            
        return np.array(arr)


class BackendManager:
    """Manager for selecting and using math backends."""
    
    _backends = {
        "python": PythonBackend,
        "numpy": NumpyBackend,
        "cupy": CupyBackend,
        "jax": JaxBackend
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
    def get_best_available_backend(cls) -> Type[Backend]:
        """Get the best available backend based on system capabilities."""
        if HAS_CUPY:
            return CupyBackend
        elif HAS_JAX:
            return JaxBackend
        elif HAS_NUMPY:
            return NumpyBackend
        else:
            return PythonBackend
    
    @classmethod
    def use_best_available(cls) -> None:
        """Switch to the best available backend."""
        backend = cls.get_best_available_backend()
        cls.set_backend(backend.name)
    
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
        try:
            return cls._current_backend.to_numpy(arr)
        except BackendError:
            # If NumPy isn't available, just return the array as is
            warnings.warn("NumPy not available for conversion, returning original array")
            return arr


# Initialize with the best available backend
BackendManager.use_best_available()


def get_math_functions():
    """Get common math functions from the current backend.
    
    Returns:
        Dictionary of math functions and constants from the current backend.
    """
    backend = BackendManager.get_backend()
    return {
        'sin': backend.sin,
        'cos': backend.cos,
        'tan': backend.tan,
        'sqrt': backend.sqrt,
        'log': backend.log,
        'exp': backend.exp,
        'pow': backend.pow,
        'pi': backend.pi,
        'e': backend.e
    }