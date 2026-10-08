"""
Intelligent spectrum allocation for 5G/6G networks.

A 3GPP-grounded cell simulator, the classical schedulers deployed today, and a
reinforcement-learning allocator that is compared against them under strict
information parity (see ``tests/test_information_parity.py``).
"""

from .config import SimConfig
from .env import Observation, SpectrumEnv

__all__ = ["SimConfig", "SpectrumEnv", "Observation"]
