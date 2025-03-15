# SplinalTap Unit Tests

This directory contains comprehensive unit tests for the SplinalTap library.

## Running Tests

You can run all tests using the test runner:

```bash
python -m splinaltap.unittest.test_runner
```

For verbose output:

```bash
python -m splinaltap.unittest.test_runner --verbose
```

To run specific test types:

```bash
# Run core tests only
python -m splinaltap.unittest.test_runner --test-type core

# Run API tests only
python -m splinaltap.unittest.test_runner --test-type api

# Run CLI tests only
python -m splinaltap.unittest.test_runner --test-type cli
```

To run a specific test file:

```bash
python -m splinaltap.unittest.test_runner --pattern test_core_keyframe.py
```

## Test Organization

The tests are organized into three main categories:

1. **Core Tests** (`test_core_*.py`): Tests for core objects and functionality
   - Instantiation of all core objects
   - Basic functionality of core classes
   - Testing all interpolation methods work as expected

2. **API Tests** (`test_api_*.py`): Tests for Python API functionality
   - Comprehensive tests of all API features
   - Tests for error handling and edge cases
   - Tests for all possible user workflows

3. **CLI Tests** (`test_cli_*.py`): Tests for command-line interface
   - Testing command-line argument parsing
   - Testing all CLI commands and options
   - Testing CLI input/output formats

## Adding New Tests

1. Create a new Python file with the naming pattern `test_{type}_{feature}.py` where:
   - `{type}` is one of: `core`, `api`, or `cli`
   - `{feature}` is the specific feature being tested

2. Use the standard `unittest` framework:
   ```python
   import unittest
   
   class TestFeature(unittest.TestCase):
       def test_something(self):
           self.assertTrue(True)
   ```

3. Run the tests using the test runner to make sure they're discovered correctly