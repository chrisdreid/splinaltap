#!/usr/bin/env python3
"""Tests for expression evaluation in SplinalTap."""

import unittest
import math
from splinaltap import ExpressionEvaluator, Channel, KeyframeSolver
from splinaltap.backends import BackendManager

class TestExpressionEvaluation(unittest.TestCase):
    """Test expression evaluation functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Reset backend to default for consistent testing
        BackendManager.set_backend("python")
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_math_expressions(self):
        """Test basic mathematical expressions."""
        # Test simple arithmetic
        self.assertEqual(self.evaluator.evaluate("1 + 2"), 3)
        self.assertEqual(self.evaluator.evaluate("3 - 2"), 1)
        self.assertEqual(self.evaluator.evaluate("2 * 3"), 6)
        self.assertEqual(self.evaluator.evaluate("6 / 3"), 2)
        self.assertEqual(self.evaluator.evaluate("2 ** 3"), 8)  # Exponentiation
        self.assertEqual(self.evaluator.evaluate("10 % 3"), 1)  # Modulo
        
        # Test operator precedence
        self.assertEqual(self.evaluator.evaluate("1 + 2 * 3"), 7)
        self.assertEqual(self.evaluator.evaluate("(1 + 2) * 3"), 9)
        
        # Test power notation
        self.assertEqual(self.evaluator.evaluate("2^3"), 8)  # ^ should be converted to **
    
    def test_variables(self):
        """Test expressions with variables."""
        # Define variables
        variables = {
            "x": 10,
            "y": 5,
            "t": 0.5
        }
        
        # Test expressions with variables
        self.assertEqual(self.evaluator.evaluate("x + y", variables), 15)
        self.assertEqual(self.evaluator.evaluate("x * y", variables), 50)
        self.assertEqual(self.evaluator.evaluate("x / y", variables), 2)
        self.assertEqual(self.evaluator.evaluate("t * 100", variables), 50)
        
        # Test with @ as alias for t
        self.assertEqual(self.evaluator.evaluate("@ * 100", variables), 50)
        
        # Test complex expressions
        self.assertEqual(self.evaluator.evaluate("x + y * t", variables), 12.5)
    
    def test_math_functions(self):
        """Test mathematical functions in expressions."""
        # Define variables for testing
        variables = {"t": math.pi/2, "@": math.pi/2}
        
        # Test trigonometric functions
        self.assertAlmostEqual(self.evaluator.evaluate("sin(t)", variables), 1.0, places=5)
        self.assertAlmostEqual(self.evaluator.evaluate("cos(t)", variables), 0.0, places=5)
        self.assertAlmostEqual(self.evaluator.evaluate("tan(pi/4)"), 1.0, places=5)
        
        # Test inverse trigonometric functions
        self.assertAlmostEqual(self.evaluator.evaluate("asin(1)"), math.pi/2, places=5)
        self.assertAlmostEqual(self.evaluator.evaluate("acos(0)"), math.pi/2, places=5)
        self.assertAlmostEqual(self.evaluator.evaluate("atan(1)"), math.pi/4, places=5)
        
        # Test logarithmic functions
        self.assertAlmostEqual(self.evaluator.evaluate("log(100)"), 2.0, places=5)  # Base 10
        self.assertAlmostEqual(self.evaluator.evaluate("ln(e)"), 1.0, places=5)  # Natural log
        
        # Test other functions
        self.assertEqual(self.evaluator.evaluate("abs(-5)"), 5)
        self.assertEqual(self.evaluator.evaluate("floor(4.7)"), 4)
        self.assertEqual(self.evaluator.evaluate("ceil(4.1)"), 5)
        self.assertEqual(self.evaluator.evaluate("round(4.5)"), 5)
        self.assertAlmostEqual(self.evaluator.evaluate("sqrt(16)"), 4.0, places=5)
    
    def test_complex_expressions(self):
        """Test more complex expressions."""
        # Define variables
        variables = {"t": 0.5, "@": 0.5, "pi": math.pi}
        
        # Test complex expressions
        expression1 = "sin(t * pi)"
        self.assertAlmostEqual(self.evaluator.evaluate(expression1, variables), 1.0, places=5)
        
        expression2 = "cos(2 * pi * t) * 10 + 10"
        self.assertAlmostEqual(self.evaluator.evaluate(expression2, variables), 0.0, places=5)
        
        expression3 = "sqrt(t) * 10"
        self.assertAlmostEqual(self.evaluator.evaluate(expression3, variables), 7.07, places=2)
    
    def test_expression_security(self):
        """Test that expressions are properly sanitized for security."""
        # Attempt potentially harmful expressions
        with self.assertRaises(Exception):
            self.evaluator.evaluate("__import__('os').system('ls')")
        
        with self.assertRaises(Exception):
            self.evaluator.evaluate("open('/etc/passwd').read()")
    
    def test_random_functions(self):
        """Test random number generation functions."""
        # Test rand() function (0-1 range)
        for _ in range(10):
            result = self.evaluator.evaluate("rand()")
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLess(result, 1.0)
        
        # Test randint() function with single argument (0 to max)
        for _ in range(10):
            result = self.evaluator.evaluate("randint(5)")
            self.assertIsInstance(result, float)  # Python numbers, so still float
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 5.0)
            self.assertEqual(result, int(result))  # Should be an integer value
        
        # Test randint() function with range
        for _ in range(10):
            result = self.evaluator.evaluate("randint([10, 20])")
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 10.0)
            self.assertLessEqual(result, 20.0)
            self.assertEqual(result, int(result))  # Should be an integer value
    
    def test_expressions_in_keyframes(self):
        """Test expression evaluation in keyframes."""
        channel = Channel()
        
        # Add keyframes with expressions
        channel.add_keyframe(at=0.0, value="0")
        channel.add_keyframe(at=0.5, value="sin(@ * pi)")
        channel.add_keyframe(at=1.0, value="10")
        
        # Test evaluation at keyframe points
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertAlmostEqual(channel.get_value(0.5), 1.0, places=5)
        self.assertEqual(channel.get_value(1.0), 10.0)
        
        # Test evaluation at intermediate points (should interpolate correctly)
        self.assertGreater(channel.get_value(0.25), 0.0)
        self.assertLess(channel.get_value(0.25), 1.0)
    
    def test_expressions_with_solver_variables(self):
        """Test expression evaluation with solver variables."""
        solver = KeyframeSolver()
        solver.set_variable("amplitude", 10)
        solver.set_variable("frequency", 2)
        
        spline = solver.create_spline("wave")
        channel = spline.add_channel("y")
        
        # Add keyframes with expressions using solver variables
        channel.add_keyframe(at=0.0, value="0")
        channel.add_keyframe(at=0.5, value="amplitude * sin(frequency * @ * pi)")
        channel.add_keyframe(at=1.0, value="0")
        
        # Test that variables are properly used
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertAlmostEqual(channel.get_value(0.25), 10.0, places=5)  # sin(π/2) = 1.0
        self.assertAlmostEqual(channel.get_value(0.5), 0.0, places=5)    # sin(π) = 0.0
        self.assertAlmostEqual(channel.get_value(0.75), -10.0, places=5) # sin(3π/2) = -1.0
        self.assertEqual(channel.get_value(1.0), 0.0)
        
        # Change variables and test again
        solver.set_variable("amplitude", 5)
        solver.set_variable("frequency", 1)
        
        self.assertEqual(channel.get_value(0.0), 0.0)
        self.assertAlmostEqual(channel.get_value(0.25), 5.0 * math.sin(0.25 * math.pi), places=5)
        self.assertAlmostEqual(channel.get_value(0.5), 0.0, places=5)
        self.assertAlmostEqual(channel.get_value(0.75), 5.0 * math.sin(0.75 * math.pi), places=5)
        self.assertEqual(channel.get_value(1.0), 0.0)

if __name__ == "__main__":
    unittest.main()