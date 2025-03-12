#!/usr/bin/env python3
"""
Main entry point for running splinaltap from the command line.
This can be run in two ways:
    python splinaltap <args>  (assuming parent directory is in PYTHONPATH)
    splinaltap <args>         (if installed via pip)

IMPORTANT: All CLI functionality must be contained within the splinaltap directory.
No files should be created outside this directory to maintain repo integrity.
"""

import sys
import os

# This is the entry point when running as a module or directory
if __name__ == "__main__":
    try:
        # We assume the parent directory is in sys.path
        # This allows us to import from splinaltap directly
        import splinaltap.cli
        sys.exit(splinaltap.cli.main())
    except ImportError as e:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if parent directory is in sys.path
        if parent_dir not in sys.path:
            # Try to add the parent directory to sys.path
            sys.path.insert(0, parent_dir)
            
            try:
                # Try importing again
                import splinaltap.cli
                sys.exit(splinaltap.cli.main())
            except ImportError:
                print("Error: Parent directory is not in Python path")
                print(f"Python path should include: {parent_dir}")
                print(f"Current sys.path: {sys.path}")
                sys.exit(1)
        else:
            print(f"Error importing CLI: {e}")
            sys.exit(1)

