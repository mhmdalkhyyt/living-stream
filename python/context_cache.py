"""ContextCache - Thread-safe LRU cache for built contexts."""

from threading import RLock
from typing import Optional, Dict, Any
from cachetools import LRUCache

from .context import Context


class ContextCache:
    """Thread-safe LRU-style cache for built contexts.
    
    Uses cachetools.LRUCache internally with threading.RLock for thread safety.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize context cache.
        
        Args:
            max_size: Maximum number of cached contexts.
        """
        self._cache: LRUCache[int, Context] = LRUCache(maxsize=max_size)
        self._lock = RLock()
    
    def cache_context(self, slot_id: int, context: Context) -> None:
        """Cache a context for the given slot ID.
        
        If entry exists, updates it and moves to most recently used.
        
        Args:
            slot_id: The slot ID to cache under.
            context: The context to cache.
        """
        with self._lock:
            self._cache[slot_id] = context
    
    def get_cached_context(self, slot_id: int) -> Optional[Context]:
        """Retrieve cached context for the given slot ID.
        
        Moves the accessed item to most recently used.
        
        Args:
            slot_id: The slot ID to retrieve.
            
        Returns:
            The cached context, or None if not found.
        """
        with self._lock:
            if slot_id not in self._cache:
                return None
            
            # Accessing moves to most recently used in LRU
            context = self._cache[slot_id]
            return context
    
    def remove_context(self, slot_id: int) -> bool:
        """Remove cached context for the given slot ID.
        
        Args:
            slot_id: The slot ID to remove.
            
        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if slot_id in self._cache:
                del self._cache[slot_id]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached contexts."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get the number of cached contexts.
        
        Returns:
            Cache size.
        """
        with self._lock:
            return len(self._cache)
    
    def contains(self, slot_id: int) -> bool:
        """Check if a context is cached for the given slot ID.
        
        Args:
            slot_id: The slot ID to check.
            
        Returns:
            True if cached, False otherwise.
        """
        with self._lock:
            return slot_id in self._cache
    
    def get_max_size(self) -> int:
        """Get the maximum cache size.
        
        Returns:
            Maximum cache size.
        """
        return self._cache.maxsize
    
    def __contains__(self, slot_id: int) -> bool:
        """Support 'in' operator for checking cache membership."""
        return self.contains(slot_id)
    
    def __len__(self) -> int:
        """Support len() for getting cache size."""
        return self.size()
    
    def __repr__(self) -> str:
        return f"ContextCache(size={self.size()}, max_size={self.get_max_size()})"
