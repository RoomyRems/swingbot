"""Swingbot v2 research package."""

from .config import AppConfig, load_config
from .strategy import DEFAULT_RULES, generate_signal, prepare_indicators

__all__ = [
    "AppConfig",
    "DEFAULT_RULES",
    "generate_signal",
    "load_config",
    "prepare_indicators",
]

__version__ = "0.1.0"
