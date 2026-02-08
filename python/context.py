"""Context - Immutable built context containing model data."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
import numpy as np


@dataclass
class Context:
    """Immutable built context containing model data.
    
    Supports both Python lists and numpy arrays for weights.
    """
    
    weights: Union[List[float], np.ndarray] = field(default_factory=list)
    config: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def get_weights(self) -> Union[List[float], np.ndarray]:
        """Get model weights."""
        return self.weights
    
    def set_weights(self, weights: Union[List[float], np.ndarray]) -> None:
        """Set model weights."""
        self.weights = weights
    
    def get_config(self) -> Dict[str, float]:
        """Get configuration parameters."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, float]) -> None:
        """Set configuration parameters."""
        self.config = config.copy()
    
    def get_metadata(self, key: str) -> str:
        """Get metadata value by key."""
        return self.metadata.get(key, "")
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        self.metadata[key] = value
    
    def __str__(self) -> str:
        weights_count = len(self.weights) if hasattr(self.weights, '__len__') else 0
        return (f"Context(weights={weights_count}, "
                f"config_keys={list(self.config.keys())}, "
                f"metadata_keys={list(self.metadata.keys())})")
