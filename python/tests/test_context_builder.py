"""Tests for the ContextBuilder module."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from python.context_builder import (
    ContextBuilder,
    BuildTask,
    BuildResult,
    BuildPriority,
    build_context,
    async_build_context,
    build_context_batch,
    async_build_context_batch,
    rebuild_context,
    async_rebuild_context,
)
from python.context import Context
from python.context_cache import ContextCache
from python.node import Node
from python.llm_node import LLMNode
from python.cnn_node import CNNNode


class TestBuildTask:
    """Tests for the BuildTask dataclass."""

    def test_default_priority(self):
        """Test that default priority is NORMAL."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        assert task.priority == BuildPriority.NORMAL

    def test_custom_priority(self):
        """Test that custom priority is set correctly."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node, priority=BuildPriority.HIGH)
        assert task.priority == BuildPriority.HIGH

    def test_force_rebuild_default(self):
        """Test that force_rebuild defaults to False."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        assert task.force_rebuild is False

    def test_callback_default(self):
        """Test that callback defaults to None."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        assert task.callback is None

    def test_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        assert task.metadata == {}

    def test_less_than_priority_comparison(self):
        """Test that priority comparison works correctly."""
        node = LLMNode(slot_index=0)
        low_task = BuildTask(slot_id=0, node=node, priority=BuildPriority.LOW)
        high_task = BuildTask(slot_id=1, node=node, priority=BuildPriority.HIGH)
        assert low_task < high_task


class TestBuildResult:
    """Tests for the BuildResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        context = Context()
        result = BuildResult(
            slot_id=0,
            context=context,
            success=True,
            duration_ms=10.5
        )
        assert result.success is True
        assert result.context == context
        assert result.error is None
        assert result.duration_ms == 10.5

    def test_failed_result(self):
        """Test creating a failed result."""
        result = BuildResult(
            slot_id=0,
            context=None,
            success=False,
            error="Build failed"
        )
        assert result.success is False
        assert result.context is None
        assert result.error == "Build failed"

    def test_from_cache_flag(self):
        """Test from_cache flag."""
        context = Context()
        cached_result = BuildResult(
            slot_id=0,
            context=context,
            success=True,
            from_cache=True
        )
        assert cached_result.from_cache is True


class TestContextBuilder:
    """Tests for the ContextBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a ContextBuilder instance."""
        return ContextBuilder()

    @pytest.fixture
    def builder_with_cache(self):
        """Create a ContextBuilder with cache."""
        return ContextBuilder(cache=ContextCache(max_size=10))

    def test_initialization(self, builder):
        """Test builder initializes correctly."""
        assert builder._cache is None
        assert builder._max_workers == 4
        assert builder._stats['total_builds'] == 0

    def test_initialization_with_cache(self, builder_with_cache):
        """Test builder initializes with cache."""
        assert builder_with_cache._cache is not None
        assert builder_with_cache._cache.get_max_size() == 10

    def test_context_manager(self):
        """Test context manager protocol."""
        with ContextBuilder() as builder:
            assert isinstance(builder, ContextBuilder)
        # Context manager should not raise

    def test_executor_property(self, builder):
        """Test executor property creates executor lazily."""
        executor = builder.executor
        assert isinstance(executor, ThreadPoolExecutor)
        # Same executor should be returned
        assert builder.executor is executor

    def test_cache_property(self, builder_with_cache):
        """Test cache property getter and setter."""
        new_cache = ContextCache(max_size=20)
        builder_with_cache.cache = new_cache
        assert builder_with_cache.cache is new_cache

    def test_build_sync_no_cache(self, builder):
        """Test synchronous build without cache."""
        node = LLMNode(slot_index=0)
        result = builder.build_sync(node)
        
        assert result.success is True
        assert result.context is not None
        assert result.context.get_metadata("model_type") == "LLM"
        assert result.from_cache is False

    def test_build_sync_with_cache_hit(self, builder_with_cache):
        """Test build with cache hit."""
        node = LLMNode(slot_index=0)
        
        # First build - populates cache
        result1 = builder_with_cache.build_sync(node)
        assert result1.success is True
        assert result1.from_cache is False
        
        # Second build - should hit cache
        result2 = builder_with_cache.build_sync(node)
        assert result2.success is True
        assert result2.from_cache is True

    def test_build_sync_with_cache_miss(self, builder_with_cache):
        """Test build with cache miss."""
        node1 = LLMNode(slot_index=1)
        node2 = LLMNode(slot_index=2)
        
        # Build for node1
        result1 = builder_with_cache.build_sync(node1)
        assert result1.success is True
        assert result1.from_cache is False
        
        # Build for node2 - different slot, should miss cache
        result2 = builder_with_cache.build_sync(node2)
        assert result2.success is True
        assert result2.from_cache is False

    @pytest.mark.asyncio
    async def test_build_async(self, builder):
        """Test asynchronous build."""
        node = LLMNode(slot_index=0)
        result = await builder.build(node)
        
        assert result.success is True
        assert result.context is not None
        assert result.context.get_metadata("model_type") == "LLM"

    @pytest.mark.asyncio
    async def test_build_async_with_cache(self, builder_with_cache):
        """Test async build with cache."""
        node = LLMNode(slot_index=0)
        
        # First build
        result1 = await builder_with_cache.build(node)
        assert result1.success is True
        assert result1.from_cache is False
        
        # Second build - should hit cache
        result2 = await builder_with_cache.build(node)
        assert result2.success is True
        assert result2.from_cache is True

    def test_build_batch_sync(self, builder):
        """Test batch synchronous build."""
        nodes = [
            LLMNode(slot_index=0),
            LLMNode(slot_index=1),
            CNNNode(slot_index=2)
        ]
        results = builder.build_batch_sync(nodes)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.context is not None for r in results)

    @pytest.mark.asyncio
    async def test_build_batch_async(self, builder):
        """Test batch asynchronous build."""
        nodes = [
            LLMNode(slot_index=0),
            LLMNode(slot_index=1),
            CNNNode(slot_index=2)
        ]
        results = await builder.build_batch(nodes)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.context is not None for r in results)

    def test_rebuild_sync(self, builder_with_cache):
        """Test synchronous rebuild invalidates cache."""
        node = LLMNode(slot_index=0)
        
        # Initial build
        result1 = builder_with_cache.build_sync(node)
        assert result1.from_cache is False
        
        # Rebuild should invalidate cache
        result2 = builder_with_cache.rebuild_sync(slot_id=0, node=node)
        assert result2.success is True
        # After invalidation, should build fresh
        assert result2.from_cache is False

    @pytest.mark.asyncio
    async def test_rebuild_async(self, builder_with_cache):
        """Test asynchronous rebuild."""
        node = LLMNode(slot_index=0)
        
        # Initial build
        result1 = await builder_with_cache.build(node)
        assert result1.from_cache is False
        
        # Rebuild
        result2 = await builder_with_cache.rebuild(slot_id=0, node=node)
        assert result2.success is True
        assert result2.from_cache is False

    def test_build_with_custom_slot_id(self, builder):
        """Test build with custom slot ID."""
        node = LLMNode(slot_index=0)
        result = builder.build_sync(node, slot_id=42)
        
        assert result.success is True
        assert result.slot_id == 42

    def test_build_without_cache_use(self, builder_with_cache):
        """Test build that ignores cache."""
        node = LLMNode(slot_index=0)
        
        # First build
        result1 = builder_with_cache.build_sync(node)
        assert result1.from_cache is False
        
        # Second build without cache - should build fresh
        result2 = builder_with_cache.build_sync(node, use_cache=False)
        assert result2.success is True
        assert result2.from_cache is False

    def test_get_stats(self, builder_with_cache):
        """Test statistics tracking."""
        # Initial stats
        stats = builder_with_cache.get_stats()
        assert stats['total_builds'] == 0
        assert stats['cache_hits'] == 0
        
        # Build to populate stats
        node = LLMNode(slot_index=0)
        builder_with_cache.build_sync(node)
        builder_with_cache.build_sync(node)  # Cache hit
        
        stats = builder_with_cache.get_stats()
        assert stats['total_builds'] == 2
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['cache_hit_rate'] == 0.5

    def test_clear_cache(self, builder_with_cache):
        """Test cache clearing."""
        node = LLMNode(slot_index=0)
        builder_with_cache.build_sync(node)
        
        assert builder_with_cache.cache.size() == 1
        
        builder_with_cache.clear_cache()
        
        assert builder_with_cache.cache.size() == 0

    def test_repr(self, builder_with_cache):
        """Test string representation."""
        node = LLMNode(slot_index=0)
        builder_with_cache.build_sync(node)
        
        repr_str = repr(builder_with_cache)
        assert "ContextBuilder" in repr_str
        assert "builds=" in repr_str


class TestBuildTaskSubmission:
    """Tests for build task submission."""

    @pytest.fixture
    def builder(self):
        """Create a builder with cache."""
        return ContextBuilder(cache=ContextCache(max_size=10))

    def test_submit_task_sync(self, builder):
        """Test synchronous task submission."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        
        result = builder.submit_task_sync(task)
        
        assert result.success is True
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_submit_task_async(self, builder):
        """Test asynchronous task submission."""
        node = LLMNode(slot_index=0)
        task = BuildTask(slot_id=0, node=node)
        
        result = await builder.submit_task(task)
        
        assert result.success is True
        assert result.context is not None

    def test_submit_task_with_callback(self, builder):
        """Test task submission with callback."""
        node = LLMNode(slot_index=0)
        callback_results = []
        
        def callback(result):
            callback_results.append(result)
        
        task = BuildTask(
            slot_id=0,
            node=node,
            callback=callback
        )
        
        builder.submit_task_sync(task)
        
        assert len(callback_results) == 1
        assert callback_results[0].success is True

    def test_submit_task_with_force_rebuild(self, builder):
        """Test task with force_rebuild."""
        node = LLMNode(slot_index=0)
        
        # First build
        builder.build_sync(node)
        
        # Task with force rebuild
        task = BuildTask(
            slot_id=0,
            node=node,
            force_rebuild=True
        )
        
        result = builder.submit_task_sync(task)
        
        assert result.success is True
        assert result.from_cache is False

    def test_submit_task_uses_cache(self, builder):
        """Test that task submission respects cache."""
        node = LLMNode(slot_index=0)
        
        # First submission
        task1 = BuildTask(slot_id=0, node=node)
        result1 = builder.submit_task_sync(task1)
        
        # Second submission - should hit cache
        task2 = BuildTask(slot_id=0, node=node)
        result2 = builder.submit_task_sync(task2)
        
        assert result1.from_cache is False
        assert result2.from_cache is True


class TestFunctionalAPI:
    """Tests for the functional API functions."""

    def test_build_context_function(self):
        """Test build_context function."""
        node = LLMNode(slot_index=0)
        context = build_context(node)
        
        assert context is not None
        assert context.get_metadata("model_type") == "LLM"

    def test_build_context_with_cache(self):
        """Test build_context with cache."""
        cache = ContextCache(max_size=10)
        node = LLMNode(slot_index=0)
        
        context = build_context(node, cache=cache)
        
        assert context is not None
        
        # Verify it's in cache
        cached = cache.get_cached_context(0)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_async_build_context_function(self):
        """Test async_build_context function."""
        node = LLMNode(slot_index=0)
        context = await async_build_context(node)
        
        assert context is not None
        assert context.get_metadata("model_type") == "LLM"

    def test_build_context_batch_function(self):
        """Test build_context_batch function."""
        nodes = [
            LLMNode(slot_index=0),
            CNNNode(slot_index=1),
            LLMNode(slot_index=2)
        ]
        contexts = build_context_batch(nodes)
        
        assert len(contexts) == 3
        assert all(c is not None for c in contexts)

    @pytest.mark.asyncio
    async def test_async_build_context_batch_function(self):
        """Test async_build_context_batch function."""
        nodes = [
            LLMNode(slot_index=0),
            CNNNode(slot_index=1),
        ]
        contexts = await async_build_context_batch(nodes)
        
        assert len(contexts) == 2
        assert all(c is not None for c in contexts)

    def test_rebuild_context_function(self):
        """Test rebuild_context function."""
        cache = ContextCache(max_size=10)
        node = LLMNode(slot_index=0)
        
        # Initial build
        context1 = build_context(node, cache=cache)
        
        # Rebuild
        context2 = rebuild_context(slot_id=0, node=node, cache=cache)
        
        assert context2 is not None
        assert context1 is not context2  # Should be new instance

    @pytest.mark.asyncio
    async def test_async_rebuild_context_function(self):
        """Test async_rebuild_context function."""
        cache = ContextCache(max_size=10)
        node = LLMNode(slot_index=0)
        
        context = await async_rebuild_context(
            slot_id=0,
            node=node,
            cache=cache
        )
        
        assert context is not None

    def test_build_context_batch_with_max_workers(self):
        """Test batch building with custom max_workers."""
        nodes = [
            LLMNode(slot_index=i)
            for i in range(4)
        ]
        contexts = build_context_batch(nodes, max_workers=2)
        
        assert len(contexts) == 4

    def test_build_context_function_error_handling(self):
        """Test that build_context raises on failure."""
        # Create a mock node that raises an exception
        mock_node = Mock(spec=Node)
        mock_node.get_slot_index.return_value = 0
        mock_node.async_build_context.side_effect = RuntimeError("Test error")
        
        with pytest.raises(RuntimeError, match="Context build failed"):
            build_context(mock_node)


class TestContextBuilderEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def builder(self):
        """Create a builder with cache."""
        return ContextBuilder(cache=ContextCache(max_size=10))

    def test_build_sync_with_failing_node(self):
        """Test build with a node that fails."""
        builder = ContextBuilder()
        mock_node = Mock(spec=Node)
        mock_node.get_slot_index.return_value = 0
        mock_node.build_context.side_effect = RuntimeError("Simulated failure")
        
        result = builder.build_sync(mock_node)
        
        assert result.success is False
        assert result.error == "Simulated failure"
        assert result.context is None

    @pytest.mark.asyncio
    async def test_build_async_with_failing_node(self):
        """Test async build with a node that fails."""
        builder = ContextBuilder()
        mock_node = Mock(spec=Node)
        mock_node.get_slot_index.return_value = 0
        mock_node.async_build_context.side_effect = RuntimeError("Async failure")
        
        result = await builder.build(mock_node)
        
        assert result.success is False
        assert result.error == "Async failure"

    def test_build_batch_sync_empty_list(self):
        """Test batch building empty list."""
        builder = ContextBuilder()
        results = builder.build_batch_sync([])
        
        assert results == []

    @pytest.mark.asyncio
    async def test_build_batch_async_empty_list(self):
        """Test async batch building empty list."""
        builder = ContextBuilder()
        results = await builder.build_batch([])
        
        assert results == []

    def test_rebuild_batch_sync(self, builder):
        """Test batch rebuild synchronous."""
        nodes = [LLMNode(slot_index=i) for i in range(3)]
        
        # Populate cache first
        builder.build_batch_sync(nodes)
        
        # Rebuild all
        results = []
        for i, node in enumerate(nodes):
            result = builder.rebuild_sync(slot_id=i, node=node)
            results.append(result)
        
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_rebuild_batch_async(self, builder):
        """Test batch rebuild asynchronous."""
        nodes = [LLMNode(slot_index=i) for i in range(3)]
        
        # Populate cache first
        await builder.build_batch(nodes)
        
        # Rebuild all
        results = []
        for i, node in enumerate(nodes):
            result = await builder.rebuild(slot_id=i, node=node)
            results.append(result)
        
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_custom_max_workers(self):
        """Test builder with custom max_workers."""
        builder = ContextBuilder(max_workers=2)
        assert builder._max_workers == 2
        
        nodes = [LLMNode(slot_index=i) for i in range(4)]
        results = builder.build_batch_sync(nodes, max_workers=3)
        
        assert len(results) == 4
        
        builder.close()

    def test_close_method(self):
        """Test builder close method."""
        builder = ContextBuilder()
        executor = builder.executor
        builder.close()
        
        # Close should not raise
        assert builder._executor is None

    def test_executor_shutdown_on_del(self):
        """Test executor is shutdown when builder is deleted."""
        import weakref
        import gc
        
        builder = ContextBuilder()
        executor = builder.executor
        ref = weakref.ref(builder)
        
        del builder
        gc.collect()
        
        # Reference should be dead
        assert ref() is None


class TestBuildPriority:
    """Tests for BuildPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert BuildPriority.LOW.value < BuildPriority.NORMAL.value
        assert BuildPriority.NORMAL.value < BuildPriority.HIGH.value
        assert BuildPriority.HIGH.value < BuildPriority.CRITICAL.value

    def test_all_priorities_exist(self):
        """Test all expected priorities exist."""
        assert BuildPriority.LOW == BuildPriority.LOW
        assert BuildPriority.NORMAL == BuildPriority.NORMAL
        assert BuildPriority.HIGH == BuildPriority.HIGH
        assert BuildPriority.CRITICAL == BuildPriority.CRITICAL
