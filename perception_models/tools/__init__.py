"""
Mobile deployment tools for Perception Encoder models.

This module provides utilities for:
- Converting PE models to CoreML format
- Benchmarking mobile performance
- Optimizing models for iOS deployment
"""

from .convert import CoreMLConverter

__all__ = ["CoreMLConverter"]
