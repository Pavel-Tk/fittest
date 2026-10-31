"""Fittest: a collective that decides which MLIP to trust, where, and repairs it where
none can be. See docs/DESIGN.md for the contract between modules.

This package root deliberately imports nothing beyond the standard library: model-server
processes run in a minimal (numpy + torch) environment and import only
``fittest.data`` / ``fittest.model_server``.
"""

__version__ = "0.1.0"
