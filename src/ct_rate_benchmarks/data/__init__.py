# src/ct_rate_benchmarks/data/__init__.py

"""
Data Package Initialization.

This package contains modules related to data loading, processing,
and dataset definitions for the CT-RATE feature benchmarks.
"""

# Expose the main Dataset class for easier access
from .dataset import FeatureDataset

__all__ = ["FeatureDataset"]