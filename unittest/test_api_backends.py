#!/usr/bin/env python3
"""Tests for backend management and switching in SplinalTap."""

import unittest
from splinaltap import KeyframeSolver, Channel
from splinaltap.backends import BackendManager

class TestBackends(unittest.TestCase):
    """Test backend management and switching."""
    
    def setUp(self):
        """Set up test case."""
        # Reset to default backend
        BackendManager.set_backend("python")
    
    def test_available_backends(self):
        """Test getting available backends."""
        backends = BackendManager.available_backends()
        
        # Python backend should always be available
        self.assertIn("python", backends)
        
        # Other backends may include numpy, jax, etc. depending on installation
        # We don't test for specific backends beyond python since they're optional
    
    def test_get_current_backend(self):
        """Test getting the current backend."""
        # Get current backend (should be python from setUp)
        backend = BackendManager.get_backend()
        
        # Verify properties
        self.assertEqual(backend.name, "python")
        self.assertEqual(backend.supports_gpu, False)  # Python backend doesn't support GPU
        
        # Math functions should include basic operations
        math_funcs = backend.get_math_functions()
        self.assertIn("sin", math_funcs)
        self.assertIn("cos", math_funcs)
        self.assertIn("sqrt", math_funcs)
    
    def test_set_backend(self):
        """Test setting the backend."""
        # Get available backends
        backends = BackendManager.available_backends()
        
        # Test switching to each available backend
        for backend_name in backends:
            # Set the backend
            BackendManager.set_backend(backend_name)
            
            # Verify that the backend was set
            current_backend = BackendManager.get_backend()
            self.assertEqual(current_backend.name, backend_name)
            
            # Test basic functionality with this backend
            channel = Channel()
            channel.add_keyframe(at=0.0, value=0.0)
            channel.add_keyframe(at=1.0, value=10.0)
            
            # Get interpolated value
            value = channel.get_value(0.5)
            self.assertEqual(value, 5.0)
        
        # Test setting to an invalid backend
        with self.assertRaises(ValueError):
            BackendManager.set_backend("nonexistent_backend")
    
    def test_use_best_available(self):
        """Test using the best available backend."""
        # Use best available
        BackendManager.use_best_available()
        
        # Get the selected backend
        backend = BackendManager.get_backend()
        
        # Verify that a backend was selected
        self.assertIsNotNone(backend)
        self.assertIsNotNone(backend.name)
        
        # Note: We can't easily predict which backend will be selected as "best"
        # since it depends on the test environment, but we can test functionality
        
        # Test basic functionality with the selected backend
        channel = Channel()
        channel.add_keyframe(at=0.0, value=0.0)
        channel.add_keyframe(at=1.0, value=10.0)
        
        # Get interpolated value
        value = channel.get_value(0.5)
        self.assertEqual(value, 5.0)
    
    def test_consistency_across_backends(self):
        """Test that results are consistent across different backends."""
        # Get available backends
        backends = BackendManager.available_backends()
        
        # Skip test if there's only one backend
        if len(backends) < 2:
            self.skipTest("Need at least two backends to test consistency")
        
        # Create a test channel with complex expressions
        channel = Channel()
        channel.add_keyframe(at=0.0, value="sin(0)")
        channel.add_keyframe(at=0.5, value="sin(pi/2)")
        channel.add_keyframe(at=1.0, value="sin(pi)")
        
        # Get results with each backend
        results = {}
        
        for backend_name in backends:
            BackendManager.set_backend(backend_name)
            
            # Sample at various points
            sample_points = [0.0, 0.25, 0.5, 0.75, 1.0]
            values = channel.sample(sample_points)
            results[backend_name] = values
        
        # Compare results across backends
        reference_backend = next(iter(backends))
        reference_values = results[reference_backend]
        
        for backend_name, values in results.items():
            if backend_name == reference_backend:
                continue
                
            for i, (ref_val, val) in enumerate(zip(reference_values, values)):
                # Allow for small floating-point differences between backends
                self.assertAlmostEqual(ref_val, val, places=5, 
                                      msg=f"Inconsistent result at index {i} between "
                                          f"{reference_backend} and {backend_name}")
    
    def test_backend_specific_features(self):
        """Test backend-specific features or optimizations."""
        # This test focuses on the vectorized evaluation capability
        # that may be present in optimized backends like numpy or jax
        
        # Create a large solver with many splines and channels
        solver = KeyframeSolver()
        for i in range(10):
            spline = solver.create_spline(f"spline{i}")
            for j in range(10):
                channel = spline.add_channel(f"channel{j}")
                channel.add_keyframe(at=0.0, value=0.0)
                channel.add_keyframe(at=1.0, value=float(i + j))
        
        # Define many sample points
        sample_points = [i / 100 for i in range(101)]
        
        # Test with each available backend
        backends = BackendManager.available_backends()
        for backend_name in backends:
            BackendManager.set_backend(backend_name)
            
            # Time the evaluation (we're not actually testing speed here, just functionality)
            results = solver.solve_multiple(sample_points)
            
            # Verify that the results have correct structure and size
            self.assertEqual(len(results), len(sample_points))
            
            # Verify a few sample values
            for t_idx, t in enumerate([0.0, 0.5, 1.0]):
                if t in sample_points:
                    result_idx = sample_points.index(t)
                    result = results[result_idx]
                    
                    # Check a few specific values
                    for i in range(3):
                        for j in range(3):
                            spline_name = f"spline{i}"
                            channel_name = f"channel{j}"
                            expected_value = float(i + j) * t
                            
                            self.assertIn(spline_name, result)
                            self.assertIn(channel_name, result[spline_name])
                            self.assertAlmostEqual(result[spline_name][channel_name], expected_value, 
                                                  places=5)

if __name__ == "__main__":
    unittest.main()