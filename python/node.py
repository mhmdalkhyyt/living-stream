"""Node - Abstract base class for all model nodes."""

from abc import ABC, abstractmethod
from typing import Optional, Awaitable, Set
import asyncio

from .context import Context


class Node(ABC):
    """Abstract base class for all model nodes.
    
    Provides interface for building contexts from model slots.
    Supports parent-child hierarchy and group memberships.
    """
    
    def __init__(self, slot_index: int):
        """Initialize node with slot index.
        
        Args:
            slot_index: The slot index for this node.
        """
        self._slot_index = slot_index
        self._parent_slot: Optional[int] = None
        self._groups: Set[str] = set()
    
    # ============= Parent Hierarchy Methods =============
    
    def set_parent(self, parent_slot: Optional[int]) -> None:
        """Set the parent slot for this node.
        
        Args:
            parent_slot: The parent slot index, or None for root
        """
        self._parent_slot = parent_slot
    
    def get_parent(self) -> Optional[int]:
        """Get the parent slot for this node.
        
        Returns:
            Parent slot index, or None if root/not set
        """
        return self._parent_slot
    
    # ============= Group Membership Methods =============
    
    def add_to_group(self, group_name: str) -> bool:
        """Add this node to a group.
        
        Args:
            group_name: The name of the group
            
        Returns:
            True if added, False if already in group
        """
        if group_name in self._groups:
            return False
        self._groups.add(group_name)
        return True
    
    def remove_from_group(self, group_name: str) -> bool:
        """Remove this node from a group.
        
        Args:
            group_name: The name of the group
            
        Returns:
            True if removed, False if not in group
        """
        if group_name not in self._groups:
            return False
        self._groups.remove(group_name)
        return True
    
    def get_groups(self) -> Set[str]:
        """Get all groups this node belongs to.
        
        Returns:
            Set of group names
        """
        return self._groups.copy()
    
    def is_in_group(self, group_name: str) -> bool:
        """Check if this node is in a group.
        
        Args:
            group_name: The name of the group
            
        Returns:
            True if in group
        """
        return group_name in self._groups
    
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
