"""Ensures the repo root is on sys.path so `import src...` works regardless
of how pytest is invoked."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
