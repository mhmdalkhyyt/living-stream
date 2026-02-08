"""Node - Abstract base class for all model nodes."""

from abc import ABC, abstractmethod
from typing import Optional, Awaitable
import asyncio

from .context import Context


class Node(ABC):
    """Abstract base class for all model nodes.
    
    Provides interface for building contexts from model slots.
    """
    
    def __init__(self, slot_index: int):
        """Initialize node with slot index.
        
        Args:
            slot_index: The slot index for this node.
        """
        self._slot_index = slot_index
    
    @abstractmethod
    def build_context(self) -> Context:
        """Build context synchronously.
        
        Returns:
            A new Context instance.
        """
        pass
    
    @abstractmethod
    async def async_build_context(self) -> Context:
        """Build context asynchronously.
        
        Returns:
            A new Context instance.
        """
        pass
    
    def get_slot_index(self) -> int:
        """Get the slot index.
        
        Returns:
            The slot index.
        """
        return self._slot_index
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(slot_index={self._slot_index})"
