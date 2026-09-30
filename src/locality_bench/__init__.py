"""CPU image-editing evaluation. Importing never loads generation or judge weights."""
__version__ = "0.1.0"

from .core import MetricConfig, evaluate_pair
from .batch import aggregate, evaluate_manifest, validate_manifest

__all__ = ["MetricConfig", "evaluate_pair", "evaluate_manifest", "validate_manifest", "aggregate"]
