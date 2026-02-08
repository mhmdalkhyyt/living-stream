"""Tests for the ContextCache class."""

import pytest
import threading
import time
from python.context_cache import ContextCache
from python.context import Context


class TestContextCacheBasicOperations:
    """Tests for basic cache operations."""

    def test_cache_context_adds_context(self, empty_cache):
        """Test that cache_context adds a context."""
        context = Context()
        empty_cache.cache_context(1, context)
        assert 1 in empty_cache

    def test_get_cached_context_retrieves_context(self, empty_cache):
        """Test that get_cached_context retrieves a cached context."""
        context = Context()
        empty_cache.cache_context(1, context)
        retrieved = empty_cache.get_cached_context(1)
        assert retrieved is context

    def test_get_nonexistent_context_returns_none(self, empty_cache):
        """Test that get_cached_context returns None for missing key."""
        result = empty_cache.get_cached_context(999)
        assert result is None

    def test_remove_context_removes_existing(self, empty_cache):
        """Test that remove_context removes an existing context."""
        context = Context()
        empty_cache.cache_context(1, context)
        assert empty_cache.remove_context(1) is True
        assert 1 not in empty_cache

    def test_remove_context_returns_false_for_missing(self, empty_cache):
        """Test that remove_context returns False for missing key."""
        result = empty_cache.remove_context(999)
        assert result is False

    def test_clear_removes_all(self, populated_cache):
        """Test that clear removes all cached contexts."""
        populated_cache.clear()
        assert populated_cache.size() == 0
        assert populated_cache.contains(0) is False
        assert populated_cache.contains(1) is False


class TestContextCacheContains:
    """Tests for contains functionality."""

    def test_contains_returns_true_for_cached(self, populated_cache):
        """Test contains returns True for cached contexts."""
        assert populated_cache.contains(0) is True
        assert populated_cache.contains(1) is True

    def test_contains_returns_false_for_missing(self, empty_cache):
        """Test contains returns False for missing contexts."""
        assert empty_cache.contains(999) is False

    def test_dunder_contains_works(self, populated_cache):
        """Test __contains__ operator works."""
        assert 0 in populated_cache
        assert 1 in populated_cache
        assert 999 not in populated_cache


class TestContextCacheSize:
    """Tests for cache size functionality."""

    def test_size_returns_correct_count(self, populated_cache):
        """Test size returns correct count."""
        assert populated_cache.size() == 2

    def test_size_empty_cache_is_zero(self, empty_cache):
        """Test size returns 0 for empty cache."""
        assert empty_cache.size() == 0

    def test_dunder_len_works(self, populated_cache):
        """Test __len__ operator works."""
        assert len(populated_cache) == 2


class TestContextCacheMaxSize:
    """Tests for max size functionality."""

    def test_get_max_size_returns_configured(self, empty_cache):
        """Test get_max_size returns configured max size."""
        assert empty_cache.get_max_size() == 10

    def test_cache_respects_max_size(self):
        """Test that cache respects max size by evicting old entries."""
        cache = ContextCache(max_size=3)
        
        # Add more than max_size entries
        for i in range(5):
            cache.cache_context(i, Context())
        
        # Should only have 3 entries (oldest evicted)
        assert cache.size() == 3
        # The oldest (0, 1) should be evicted
        assert 0 not in cache
        assert 1 not in cache

    def test_cache_update_moves_to_recent(self):
        """Test that updating a context moves it to most recent."""
        cache = ContextCache(max_size=5)
        
        # Add 3 contexts
        for i in range(3):
            cache.cache_context(i, Context())
        
        # Update context 0
        cache.cache_context(0, Context())
        
        # Add 2 more to force eviction of oldest
        cache.cache_context(3, Context())
        cache.cache_context(4, Context())
        
        # Cache should be full (5 items)
        assert cache.size() == 5
        
        # Context 0 should still be present (it was updated)
        assert 0 in cache
        # Context 1 and 2 should still be present (3,4 were added after update to 0)
        # LRU order is now: 1, 2, 3, 4, 0 -> eviction happens from front
        assert 1 in cache
        assert 2 in cache


class TestContextCacheThreadSafety:
    """Tests for thread safety of cache operations."""

    def test_concurrent_cache_access(self):
        """Test that concurrent cache access doesn't crash."""
        cache = ContextCache(max_size=100)
        errors = []
        
        def writer(slot_id):
            try:
                for _ in range(50):
                    context = Context()
                    cache.cache_context(slot_id, context)
            except Exception as e:
                errors.append(e)
        
        def reader(slot_id):
            try:
                for _ in range(50):
                    cache.get_cached_context(slot_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t1 = threading.Thread(target=writer, args=(i,))
            t2 = threading.Thread(target=reader, args=(i,))
            threads.extend([t1, t2])
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_clear_and_access(self):
        """Test concurrent clear and get operations."""
        cache = ContextCache(max_size=50)
        
        # Populate cache
        for i in range(10):
            cache.cache_context(i, Context())
        
        errors = []
        
        def clear_task():
            try:
                for _ in range(20):
                    cache.clear()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def read_task():
            try:
                for _ in range(20):
                    for i in range(10):
                        cache.get_cached_context(i)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=clear_task),
            threading.Thread(target=read_task),
            threading.Thread(target=read_task),
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestContextCacheRepr:
    """Tests for __repr__ method."""

    def test_repr_includes_size(self, populated_cache):
        """Test __repr__ includes size."""
        result = repr(populated_cache)
        assert "size=2" in result

    def test_repr_includes_max_size(self, empty_cache):
        """Test __repr__ includes max size."""
        result = repr(empty_cache)
        assert "max_size=10" in result
