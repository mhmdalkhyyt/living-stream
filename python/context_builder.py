"""ContextBuilder - Decoupled context building system.

This module provides a context building system that is decoupled from nodes
and can run on its own threads. It consumes async cache reading from nodes
and provides callable functions for building and rebuilding contexts.

Key Features:
- Thread-safe operations with ThreadPoolExecutor and asyncio
- Functional API for simple usage patterns
- Batch context building with configurable parallelism
- Cache integration with invalidation support
- Priority-based task scheduling
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from .context import Context
from .context_cache import ContextCache
from .node import Node


class BuildPriority(Enum):
    """Priority levels for build tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BuildTask:
    """Represents a build or rebuild instruction.
    
    Attributes:
        slot_id: The slot identifier for this task.
        node: The node to build context from.
        priority: Task priority level.
        force_rebuild: If True, ignores cached context and rebuilds.
        callback: Optional callback to invoke after build completes.
        metadata: Additional metadata for the task.
    """
    slot_id: int
    node: Node
    priority: BuildPriority = BuildPriority.NORMAL
    force_rebuild: bool = False
    callback: Optional[Callable[['BuildResult'], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'BuildTask') -> bool:
        """Compare tasks by priority for scheduling."""
        return self.priority.value < other.priority.value


@dataclass
class BuildResult:
    """Result of a build operation.
    
    Attributes:
        slot_id: The slot identifier.
        context: The built context, or None if failed.
        success: Whether the build succeeded.
        error: Error message if failed.
        duration_ms: Duration in milliseconds.
        from_cache: Whether the context was retrieved from cache.
    """
    slot_id: int
    context: Optional[Context]
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    from_cache: bool = False


class ContextBuilder:
    """Decoupled context building system with thread and async support.
    
    This class provides a flexible interface for building contexts from nodes,
    with support for threading, async operations, caching, and batch processing.
    
    Usage:
        # Async usage
        builder = ContextBuilder()
        result = await builder.build(node)
        
        # Batch building
        results = await builder.build_batch([node1, node2, node3])
        
        # Rebuild with cache invalidation
        result = await builder.rebuild(slot_id=5, node=node)
        
        # Sync usage (blocking)
        result = builder.build_sync(node)
    """
    
    def __init__(
        self,
        cache: Optional[ContextCache] = None,
        max_workers: int = 4,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """Initialize the context builder.
        
        Args:
            cache: Optional ContextCache for caching built contexts.
            max_workers: Maximum number of worker threads for sync operations.
            executor: Optional pre-configured ThreadPoolExecutor.
        """
        self._cache = cache
        self._max_workers = max_workers
        self._executor = executor
        self._own_executor: bool = executor is None
        self._lock = threading.RLock()
        self._active_tasks: Dict[int, 'BuildTask'] = {}
        self._stats: Dict[str, Any] = {
            'total_builds': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_build_time_ms': 0.0
        }
    
    def __del__(self):
        """Cleanup executor if we own it."""
        if self._own_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
    
    def __enter__(self) -> 'ContextBuilder':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def close(self) -> None:
        """Close the builder and cleanup resources."""
        if self._own_executor and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="ContextBuilder"
            )
        return self._executor
    
    @property
    def cache(self) -> Optional[ContextCache]:
        """Get the current cache."""
        return self._cache
    
    @cache.setter
    def cache(self, cache: Optional[ContextCache]) -> None:
        """Set the cache."""
        with self._lock:
            self._cache = cache
    
    def _get_cached_context(self, slot_id: int) -> Tuple[Optional[Context], bool]:
        """Get context from cache if available.
        
        Args:
            slot_id: The slot identifier.
            
        Returns:
            Tuple of (context, was_cached).
        """
        if self._cache is None:
            return None, False
        
        with self._lock:
            if slot_id in self._cache:
                self._stats['cache_hits'] += 1
                return self._cache.get_cached_context(slot_id), True
        
        self._stats['cache_misses'] += 1
        return None, False
    
    def _cache_context(self, slot_id: int, context: Context) -> None:
        """Cache a context.
        
        Args:
            slot_id: The slot identifier.
            context: The context to cache.
        """
        if self._cache is not None:
            self._cache.cache_context(slot_id, context)
    
    def _invalidate_cache(self, slot_id: int) -> bool:
        """Invalidate cached context for a slot.
        
        Args:
            slot_id: The slot identifier.
            
        Returns:
            True if cache was invalidated, False if not cached.
        """
        if self._cache is None:
            return False
        
        with self._lock:
            return self._cache.remove_context(slot_id)
    
    def _track_build_start(self, slot_id: int) -> float:
        """Track build start for statistics."""
        with self._lock:
            self._stats['total_builds'] += 1
            return __import__('time').time()
    
    def _track_build_end(self, start_time: float) -> None:
        """Track build end for statistics."""
        end_time = __import__('time').time()
        duration_ms = (end_time - start_time) * 1000
        with self._lock:
            self._stats['total_build_time_ms'] += duration_ms
    
    def _do_build(self, node: Node, use_async: bool = True) -> Context:
        """Perform the actual context building.
        
        Args:
            node: The node to build from.
            use_async: Whether to use async method.
            
        Returns:
            Built context.
        """
        if use_async:
            # Use async method with event loop in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(node.async_build_context())
            finally:
                loop.close()
        else:
            return node.build_context()
    
    # ============= Async Build Methods =============
    
    async def build(
        self,
        node: Node,
        slot_id: Optional[int] = None,
        use_cache: bool = True
    ) -> BuildResult:
        """Build context for a single node asynchronously.
        
        Args:
            node: The node to build context from.
            slot_id: Optional slot ID (defaults to node's slot).
            use_cache: Whether to check/use cache.
            
        Returns:
            BuildResult with the built context.
        """
        actual_slot_id = slot_id if slot_id is not None else node.get_slot_index()
        start_time = self._track_build_start(actual_slot_id)
        
        # Check cache first
        if use_cache:
            context, from_cache = self._get_cached_context(actual_slot_id)
            if from_cache and context is not None:
                return BuildResult(
                    slot_id=actual_slot_id,
                    context=context,
                    success=True,
                    duration_ms=0.0,
                    from_cache=True
                )
        
        # Build context using async method
        try:
            context = await node.async_build_context()
            
            # Cache if available
            if use_cache and self._cache is not None:
                self._cache_context(actual_slot_id, context)
            
            self._track_build_end(start_time)
            
            return BuildResult(
                slot_id=actual_slot_id,
                context=context,
                success=True,
                duration_ms=(__import__('time').time() - start_time) * 1000
            )
        except Exception as e:
            return BuildResult(
                slot_id=actual_slot_id,
                context=None,
                success=False,
                error=str(e)
            )
    
    async def build_batch(
        self,
        nodes: List[Node],
        slot_ids: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        use_cache: bool = True
    ) -> List[BuildResult]:
        """Build contexts for multiple nodes asynchronously.
        
        Uses asyncio with semaphore for controlled parallelism.
        
        Args:
            nodes: List of nodes to build contexts from.
            slot_ids: Optional list of slot IDs (defaults to node slots).
            max_workers: Maximum concurrent tasks (defaults to self.max_workers).
            use_cache: Whether to check/use cache.
            
        Returns:
            List of BuildResult objects.
        """
        if len(nodes) == 0:
            return []
        
        if slot_ids is None:
            slot_ids = [n.get_slot_index() for n in nodes]
        
        workers = max_workers if max_workers is not None else self._max_workers
        semaphore = asyncio.Semaphore(workers)
        
        async def build_with_semaphore(
            node: Node,
            slot_id: int
        ) -> BuildResult:
            async with semaphore:
                return await self.build(
                    node,
                    slot_id=slot_id,
                    use_cache=use_cache
                )
        
        tasks = [
            build_with_semaphore(node, sid)
            for node, sid in zip(nodes, slot_ids)
        ]
        
        return await asyncio.gather(*tasks)
    
    async def rebuild(
        self,
        slot_id: int,
        node: Node,
        use_cache: bool = True
    ) -> BuildResult:
        """Rebuild context for a node, invalidating cache first.
        
        Args:
            slot_id: The slot identifier.
            node: The node to rebuild context from.
            use_cache: Whether to cache the result.
            
        Returns:
            BuildResult with the rebuilt context.
        """
        # Invalidate cache first
        self._invalidate_cache(slot_id)
        
        # Build fresh context
        return await self.build(
            node,
            slot_id=slot_id,
            use_cache=use_cache
        )
    
    async def rebuild_batch(
        self,
        slot_ids: List[int],
        nodes: List[Node],
        max_workers: Optional[int] = None,
        use_cache: bool = True
    ) -> List[BuildResult]:
        """Rebuild contexts for multiple nodes.
        
        Args:
            slot_ids: List of slot identifiers.
            nodes: List of nodes to rebuild contexts from.
            max_workers: Maximum concurrent tasks.
            use_cache: Whether to cache the results.
            
        Returns:
            List of BuildResult objects.
        """
        if len(slot_ids) == 0:
            return []
        
        # Invalidate all caches first
        for sid in slot_ids:
            self._invalidate_cache(sid)
        
        # Rebuild all
        return await self.build_batch(
            nodes,
            slot_ids=slot_ids,
            max_workers=max_workers,
            use_cache=use_cache
        )
    
    # ============= Sync Build Methods =============
    
    def build_sync(
        self,
        node: Node,
        slot_id: Optional[int] = None,
        use_cache: bool = True
    ) -> BuildResult:
        """Build context for a single node synchronously.
        
        Args:
            node: The node to build context from.
            slot_id: Optional slot ID (defaults to node's slot).
            use_cache: Whether to check/use cache.
            
        Returns:
            BuildResult with the built context.
        """
        actual_slot_id = slot_id if slot_id is not None else node.get_slot_index()
        start_time = self._track_build_start(actual_slot_id)
        
        # Check cache first
        if use_cache:
            context, from_cache = self._get_cached_context(actual_slot_id)
            if from_cache and context is not None:
                return BuildResult(
                    slot_id=actual_slot_id,
                    context=context,
                    success=True,
                    duration_ms=0.0,
                    from_cache=True
                )
        
        # Build context in thread pool
        try:
            context = self.executor.submit(
                self._do_build,
                node,
                use_async=False
            ).result()
            
            # Cache if available
            if use_cache and self._cache is not None:
                self._cache_context(actual_slot_id, context)
            
            self._track_build_end(start_time)
            
            return BuildResult(
                slot_id=actual_slot_id,
                context=context,
                success=True,
                duration_ms=(__import__('time').time() - start_time) * 1000
            )
        except Exception as e:
            return BuildResult(
                slot_id=actual_slot_id,
                context=None,
                success=False,
                error=str(e)
            )
    
    def build_batch_sync(
        self,
        nodes: List[Node],
        slot_ids: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        use_cache: bool = True
    ) -> List[BuildResult]:
        """Build contexts for multiple nodes synchronously.
        
        Args:
            nodes: List of nodes to build contexts from.
            slot_ids: Optional list of slot IDs.
            max_workers: Maximum concurrent threads.
            use_cache: Whether to check/use cache.
            
        Returns:
            List of BuildResult objects.
        """
        if len(nodes) == 0:
            return []
        
        if slot_ids is None:
            slot_ids = [n.get_slot_index() for n in nodes]
        
        workers = max_workers if max_workers is not None else min(len(nodes), self._max_workers)
        
        def build_task(node: Node, slot_id: int) -> BuildResult:
            return self.build_sync(
                node,
                slot_id=slot_id,
                use_cache=use_cache
            )
        
        with self.executor:
            futures = [
                self.executor.submit(build_task, node, sid)
                for node, sid in zip(nodes, slot_ids)
            ]
            
            results = [f.result() for f in futures]
        
        return results
    
    def rebuild_sync(
        self,
        slot_id: int,
        node: Node,
        use_cache: bool = True
    ) -> BuildResult:
        """Rebuild context for a node synchronously.
        
        Args:
            slot_id: The slot identifier.
            node: The node to rebuild context from.
            use_cache: Whether to cache the result.
            
        Returns:
            BuildResult with the rebuilt context.
        """
        self._invalidate_cache(slot_id)
        return self.build_sync(
            node,
            slot_id=slot_id,
            use_cache=use_cache
        )
    
    # ============= Task Queue Methods =============
    
    async def submit_task(self, task: BuildTask) -> BuildResult:
        """Submit a build task to the queue.
        
        Args:
            task: The BuildTask to execute.
            
        Returns:
            BuildResult when the task completes.
        """
        # Check cache unless force rebuild
        if not task.force_rebuild:
            context, from_cache = self._get_cached_context(task.slot_id)
            if from_cache and context is not None:
                result = BuildResult(
                    slot_id=task.slot_id,
                    context=context,
                    success=True,
                    from_cache=True
                )
                if task.callback:
                    task.callback(result)
                return result
        
        # Track active task
        with self._lock:
            self._active_tasks[task.slot_id] = task
        
        # Build context
        result = await self.build(
            task.node,
            slot_id=task.slot_id,
            use_cache=not task.force_rebuild
        )
        
        # Cache result
        if result.success and self._cache is not None:
            self._cache_context(task.slot_id, result.context)
        
        # Invoke callback
        if task.callback:
            task.callback(result)
        
        # Remove from active tasks
        with self._lock:
            self._active_tasks.pop(task.slot_id, None)
        
        return result
    
    def submit_task_sync(self, task: BuildTask) -> BuildResult:
        """Submit a build task synchronously.
        
        Args:
            task: The BuildTask to execute.
            
        Returns:
            BuildResult when the task completes.
        """
        # Check cache unless force rebuild
        if not task.force_rebuild:
            context, from_cache = self._get_cached_context(task.slot_id)
            if from_cache and context is not None:
                result = BuildResult(
                    slot_id=task.slot_id,
                    context=context,
                    success=True,
                    from_cache=True
                )
                if task.callback:
                    task.callback(result)
                return result
        
        # Build context
        result = self.build_sync(
            task.node,
            slot_id=task.slot_id,
            use_cache=not task.force_rebuild
        )
        
        # Cache result
        if result.success and self._cache is not None:
            self._cache_context(task.slot_id, result.context)
        
        # Invoke callback
        if task.callback:
            task.callback(result)
        
        return result
    
    # ============= Statistics Methods =============
    
    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics.
        
        Returns:
            Dict with build statistics.
        """
        with self._lock:
            total = self._stats['total_builds']
            cache_hits = self._stats['cache_hits']
            cache_misses = self._stats['cache_misses']
            total_time = self._stats['total_build_time_ms']
            
            return {
                'total_builds': total,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'total_build_time_ms': total_time,
                'avg_build_time_ms': total_time / total if total > 0 else 0.0,
                'cache_hit_rate': cache_hits / (cache_hits + cache_misses) 
                                  if (cache_hits + cache_misses) > 0 else 0.0,
                'active_tasks': len(self._active_tasks),
                'max_workers': self._max_workers,
                'cache_enabled': self._cache is not None,
                'cache_size': self._cache.size() if self._cache else 0
            }
    
    def clear_cache(self) -> None:
        """Clear the associated cache."""
        if self._cache is not None:
            self._cache.clear()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ContextBuilder("
            f"builds={stats['total_builds']}, "
            f"cache_hits={stats['cache_hits']}, "
            f"cache_hit_rate={stats['cache_hit_rate']:.1%}, "
            f"active={stats['active_tasks']}"
            f")"
        )


# ============= Functional API =============

def build_context(
    node: Node,
    cache: Optional[ContextCache] = None,
    use_cache: bool = True
) -> Context:
    """Build context for a node synchronously.
    
    Simple functional API for context building.
    
    Args:
        node: The node to build context from.
        cache: Optional cache to use.
        use_cache: Whether to check/use cache.
        
    Returns:
        Built Context instance.
    """
    builder = ContextBuilder(cache=cache)
    try:
        result = builder.build_sync(node, use_cache=use_cache)
        if result.success:
            return result.context
        raise RuntimeError(f"Context build failed: {result.error}")
    finally:
        builder.close()


async def async_build_context(
    node: Node,
    cache: Optional[ContextCache] = None,
    use_cache: bool = True
) -> Context:
    """Build context for a node asynchronously.
    
    Async functional API for context building.
    
    Args:
        node: The node to build context from.
        cache: Optional cache to use.
        use_cache: Whether to check/use cache.
        
    Returns:
        Built Context instance.
    """
    builder = ContextBuilder(cache=cache)
    try:
        result = await builder.build(node, use_cache=use_cache)
        if result.success:
            return result.context
        raise RuntimeError(f"Context build failed: {result.error}")
    finally:
        builder.close()


def build_context_batch(
    nodes: List[Node],
    cache: Optional[ContextCache] = None,
    max_workers: int = 4,
    use_cache: bool = True
) -> List[Context]:
    """Build contexts for multiple nodes synchronously.
    
    Functional API for batch context building.
    
    Args:
        nodes: List of nodes to build contexts from.
        cache: Optional cache to use.
        max_workers: Maximum concurrent threads.
        use_cache: Whether to check/use cache.
        
    Returns:
        List of built Context instances.
    """
    builder = ContextBuilder(cache=cache, max_workers=max_workers)
    try:
        results = builder.build_batch_sync(nodes, use_cache=use_cache)
        contexts = []
        for result in results:
            if result.success:
                contexts.append(result.context)
            else:
                raise RuntimeError(f"Context build failed for slot {result.slot_id}: {result.error}")
        return contexts
    finally:
        builder.close()


async def async_build_context_batch(
    nodes: List[Node],
    cache: Optional[ContextCache] = None,
    max_workers: int = 4,
    use_cache: bool = True
) -> List[Context]:
    """Build contexts for multiple nodes asynchronously.
    
    Async functional API for batch context building.
    
    Args:
        nodes: List of nodes to build contexts from.
        cache: Optional cache to use.
        max_workers: Maximum concurrent tasks.
        use_cache: Whether to check/use cache.
        
    Returns:
        List of built Context instances.
    """
    builder = ContextBuilder(cache=cache, max_workers=max_workers)
    try:
        results = await builder.build_batch(nodes, use_cache=use_cache)
        contexts = []
        for result in results:
            if result.success:
                contexts.append(result.context)
            else:
                raise RuntimeError(f"Context build failed for slot {result.slot_id}: {result.error}")
        return contexts
    finally:
        builder.close()


def rebuild_context(
    slot_id: int,
    node: Node,
    cache: ContextCache,
    use_cache: bool = True
) -> Context:
    """Rebuild context for a node, invalidating cache first.
    
    Functional API for context rebuilding.
    
    Args:
        slot_id: The slot identifier.
        node: The node to rebuild context from.
        cache: The cache to invalidate and use.
        use_cache: Whether to cache the result.
        
    Returns:
        Rebuilt Context instance.
    """
    builder = ContextBuilder(cache=cache)
    try:
        result = builder.rebuild_sync(slot_id, node, use_cache=use_cache)
        if result.success:
            return result.context
        raise RuntimeError(f"Context rebuild failed: {result.error}")
    finally:
        builder.close()


async def async_rebuild_context(
    slot_id: int,
    node: Node,
    cache: ContextCache,
    use_cache: bool = True
) -> Context:
    """Rebuild context for a node asynchronously.
    
    Async functional API for context rebuilding.
    
    Args:
        slot_id: The slot identifier.
        node: The node to rebuild context from.
        cache: The cache to invalidate and use.
        use_cache: Whether to cache the result.
        
    Returns:
        Rebuilt Context instance.
    """
    builder = ContextBuilder(cache=cache)
    try:
        result = await builder.rebuild(slot_id, node, use_cache=use_cache)
        if result.success:
            return result.context
        raise RuntimeError(f"Context rebuild failed: {result.error}")
    finally:
        builder.close()
