#!/usr/bin/env python3
"""Test runner for SplinalTap unittest directory."""

import unittest
import sys
import os
import logging
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('splinaltap.unittest.runner')

def main():
    """Run the tests in the unittest directory."""
    # Parse arguments
    parser = ArgumentParser(description="Run SplinalTap unit tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pattern", "-p", default="test_*.py", help="Test file pattern")
    parser.add_argument("--test-type", "-t", choices=['core', 'api', 'cli', 'all'], default='all',
                       help="Test type to run (core, api, cli, or all)")
    args = parser.parse_args()
    
    # Find the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine test pattern based on test type
    if args.test_type != 'all':
        pattern = f"test_{args.test_type}_*.py"
    else:
        pattern = args.pattern
        
    logger.info(f"Discovering tests in {test_dir} with pattern {pattern}")
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())