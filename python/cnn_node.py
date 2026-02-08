"""CNNNode - CNN-specific node specialization."""

import random
from typing import List, Union
import numpy as np

from .node import Node
from .context import Context


class CNNNode(Node):
    """CNN-specific node implementation.
    
    Generates synthetic layer structure and weights based on slot index.
    """
    
    def __init__(self, slot_index: int, model_path: str = "models/default.bin"):
        """Initialize CNN node.
        
        Args:
            slot_index: The slot index for this node.
            model_path: Path to the model file.
        """
        super().__init__(slot_index)
        self._model_path = model_path
        self._layer_count: int = 0
        self._layer_sizes: List[int] = []
    
    def build_context(self) -> Context:
        """Build CNN context with synthetic layer weights."""
        # Create new context
        context = Context()
        
        # Generate synthetic layer structure based on slot_index
        base_layers = 4 + self.get_slot_index()
        self._layer_count = base_layers
        
        self._layer_sizes.clear()
        
        weights: List[float] = []
        
        # Generate deterministic "random" weights based on slot
        random.seed(self.get_slot_index() * 67890)
        
        layer_size = 64
        for i in range(self._layer_count):
            self._layer_sizes.append(layer_size)
            
            # Generate weights for this layer (convolution kernels + biases)
            kernel_count = layer_size * 3 * 3  # 3x3 kernels
            for j in range(kernel_count):
                weights.append(random.uniform(-0.5, 0.5))
            weights.append(random.uniform(-0.5, 0.5))  # bias
            
            # Double channels every 2 layers (except last)
            if i % 2 == 0 and i < self._layer_count - 1:
                layer_size *= 2
        
        context.set_weights(weights)
        
        # Set configuration
        config = {
            "kernel_size": 3.0,
            "stride": 1.0,
            "padding": 1.0,
            "activation": 0.0  # 0=ReLU
        }
        context.set_config(config)
        
        # Set metadata
        context.set_metadata("model_type", "CNN")
        context.set_metadata("model_path", self._model_path)
        context.set_metadata("slot_index", str(self.get_slot_index()))
        context.set_metadata("layer_count", str(self._layer_count))
        
        return context
    
    async def async_build_context(self) -> Context:
        """Build CNN context asynchronously with simulated delay."""
        import asyncio
        await asyncio.sleep(0.1)  # Simulate async loading
        return self.build_context()
    
    def load_model(self, path: str) -> None:
        """Load model from path.
        
        Args:
            path: Path to model file.
        """
        self._model_path = path
        # In a real implementation, this would load actual CNN weights
        self._layer_count = 4 + self.get_slot_index()
    
    def get_layer_count(self) -> int:
        """Get layer count.
        
        Returns:
            Number of layers.
        """
        return self._layer_count
    
    def get_layer_sizes(self) -> List[int]:
        """Get layer sizes.
        
        Returns:
            List of layer sizes.
        """
        return self._layer_sizes.copy()
    
    def __repr__(self) -> str:
        return (f"CNNNode(slot_index={self.get_slot_index()}, "
                f"model_path='{self._model_path}', "
                f"layers={self._layer_count}, "
                f"layer_sizes={self._layer_sizes})")
