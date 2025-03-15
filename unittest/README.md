# SplinalTap Unit Tests

This directory contains comprehensive unit tests for the SplinalTap library.

## Running Tests

The recommended way to run tests is using the test runner script in the project root:

```bash
# From the project root directory
python run_tests.py
```

For verbose output:

```bash
python run_tests.py -v
```

To run specific test patterns:

```bash
# Run all tests with "api" in the name
python run_tests.py api

# Run specific test file pattern
python run_tests.py file_io
```

To skip JAX tests (which may fail on some systems):

```bash
python run_tests.py --skip-jax
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

## Directory Structure

The SplinalTap test suite follows a specific directory structure:

```
./                           # Project root
└── splinaltap/              # Main source code and repository
    ├── unittest/            # Unit test code (this directory)
    │   ├── test_*.py        # Test files
    │   ├── README.md        # This file
    │   ├── input/           # Test input files for unit tests
    │   │   └── parameter_solver.json
    │   └── output/          # Test output files for unit tests
    │       ├── example-01.svg
    │       ├── theme_dark.png
    │       └── ...
    │
    ├── __init__.py
    ├── *.py                 # Source modules
    └── ...
```

For GitHub repository purposes, all test files must be contained within the splinaltap directory. The unittest/input and unittest/output directories contain the necessary files for tests to run properly.

## Adding New Tests

1. Create a new Python file with the naming pattern `test_{type}_{feature}.py` where:
   - `{type}` is one of: `core`, `api`, or `cli`
   - `{feature}` is the specific feature being tested

2. Use the standard `unittest` framework:
   ```python
   import unittest
   import os
   
   class TestFeature(unittest.TestCase):
       def setUp(self):
           # Get path to unittest directory
           self.unittest_dir = os.path.dirname(os.path.abspath(__file__))
           self.input_dir = os.path.join(self.unittest_dir, "input")
           self.output_dir = os.path.join(self.unittest_dir, "output")
           
       def test_something(self):
           self.assertTrue(True)
   ```

3. Run the tests using the test runner to make sure they're discovered correctly:
   ```bash
   python run_tests.py
   ```

4. For file paths, use proper relative paths within the splinaltap directory:
   ```python
   input_file = os.path.join(self.input_dir, "input.json")
   output_file = os.path.join(self.output_dir, "result.json")
   ```

5. **IMPORTANT**: All test input and output files must be kept within the 
   `splinaltap/unittest/input` and `splinaltap/unittest/output` directories
   to ensure they're accessible in the GitHub repository.