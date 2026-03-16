"""
packages/dssatlm/tests/conftest.py
Pytest configuration -- adds src/ to the Python path.
"""
import sys
import os

# Ensure src/ is importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

