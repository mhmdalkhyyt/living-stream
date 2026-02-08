"""Tests for PersistentStore storage layer."""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np

from python.storage import PersistentStore
from python.context import Context


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def store(temp_storage_dir):
    """Create a PersistentStore instance with temp directory."""
    store = PersistentStore(storage_dir=temp_storage_dir)
    yield store
    store.close()


class TestPersistentStoreInit:
    """Tests for PersistentStore initialization."""
    
    def test_creates_directories(self, temp_storage_dir):
        """Test that storage creates necessary directories."""
        store = PersistentStore(storage_dir=temp_storage_dir)
        try:
            assert Path(temp_storage_dir).exists()
            assert Path(temp_storage_dir, "slots").exists()
            assert Path(temp_storage_dir, "wal").exists()
            assert Path(temp_storage_dir, "index.db").exists()
        finally:
            store.close()
    
    def test_custom_cache_size(self, temp_storage_dir):
        """Test custom max cache size."""
        store = PersistentStore(storage_dir=temp_storage_dir, max_cache_size=500)
        assert store._max_cache_size == 500
        store.close()


class TestSaveLoadContext:
    """Tests for saving and loading contexts."""
    
    def test_save_and_load_simple_context(self, store):
        """Test saving and loading a basic context."""
        context = Context(
            weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            config={"temperature": 0.7, "top_p": 0.9},
            metadata={"model_type": "LLM", "version": "1.0"}
        )
        
        store.save_context(1, context)
        loaded = store.load_context(1)
        
        assert loaded is not None
        assert list(loaded.get_weights()) == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert loaded.get_config()["temperature"] == 0.7
        assert loaded.get_metadata("model_type") == "LLM"
    
    def test_save_and_load_numpy_weights(self, store):
        """Test saving and loading numpy array weights."""
        weights = np.random.randn(100).astype(np.float32)
        context = Context(weights=weights)
        
        store.save_context(2, context)
        loaded = store.load_context(2)
        
        assert loaded is not None
        assert isinstance(loaded.get_weights(), np.ndarray)
        assert np.allclose(loaded.get_weights(), weights)
    
    def test_save_and_load_2d_numpy_weights(self, store):
        """Test saving and loading 2D numpy arrays (e.g., CNN weights)."""
        weights = np.random.randn(64, 128, 3, 3).astype(np.float32)
        context = Context(
            weights=weights,
            config={"kernel_size": 3}
        )
        
        store.save_context(3, context)
        loaded = store.load_context(3)
        
        assert loaded is not None
        assert loaded.get_weights().shape == (64, 128, 3, 3)
        assert loaded.get_config()["kernel_size"] == 3
    
    def test_load_nonexistent_slot(self, store):
        """Test loading a slot that doesn't exist."""
        loaded = store.load_context(999)
        assert loaded is None
    
    def test_overwrite_existing_context(self, store):
        """Test overwriting an existing slot."""
        context1 = Context(weights=[1.0, 2.0])
        store.save_context(10, context1)
        
        context2 = Context(weights=[3.0, 4.0, 5.0])
        store.save_context(10, context2)
        
        loaded = store.load_context(10)
        assert loaded is not None
        assert list(loaded.get_weights()) == [3.0, 4.0, 5.0]


class TestContextCaching:
    """Tests for in-memory caching behavior."""
    
    def test_cache_hit_after_save(self, store):
        """Test that saved context is immediately available in cache."""
        context = Context(weights=[1.0, 2.0])
        store.save_context(100, context)
        
        # Should be in cache
        with store._cache_lock:
            assert 100 in store._cache
    
    def test_cache_eviction(self, temp_storage_dir):
        """Test cache eviction when max size exceeded."""
        store = PersistentStore(storage_dir=temp_storage_dir, max_cache_size=5)
        
        # Fill cache beyond max size
        for i in range(10):
            context = Context(weights=[float(i)])
            store.save_context(i, context)
        
        # Should have evicted some entries
        with store._cache_lock:
            assert len(store._cache) <= 5
        
        store.close()


class TestMetadataOperations:
    """Tests for metadata operations."""
    
    def test_get_metadata(self, store):
        """Test getting metadata without loading weights."""
        context = Context(
            weights=np.random.randn(50),
            config={"temperature": 0.8},
            metadata={"model_type": "CNN"}
        )
        
        store.save_context(200, context)
        metadata = store.get_metadata(200)
        
        assert metadata is not None
        assert metadata['slot_id'] == 200
        assert metadata['model_type'] == 'CNN'
        assert metadata['weight_dtype'] == 'float64'
        assert metadata['config']['temperature'] == 0.8
    
    def test_list_slots(self, store):
        """Test listing all slots."""
        # Create multiple contexts
        for i in range(5):
            context = Context(weights=[float(i)])
            store.save_context(i, context)
        
        slots = store.list_slots()
        
        assert len(slots) == 5
        assert all('slot_id' in s for s in slots)
        assert all('model_type' in s for s in slots)
    
    def test_list_slots_filter_by_type(self, store):
        """Test listing slots filtered by model type."""
        llm_context = Context(
            weights=[1.0],
            metadata={"model_type": "LLM"}
        )
        cnn_context = Context(
            weights=[2.0],
            metadata={"model_type": "CNN"}
        )
        
        store.save_context(300, llm_context)
        store.save_context(301, cnn_context)
        
        llm_slots = store.list_slots(model_type="LLM")
        cnn_slots = store.list_slots(model_type="CNN")
        
        assert len(llm_slots) == 1
        assert len(cnn_slots) == 1
        assert llm_slots[0]['model_type'] == 'LLM'
        assert cnn_slots[0]['model_type'] == 'CNN'
    
    def test_list_slots_pagination(self, store):
        """Test pagination of list_slots."""
        # Create 10 contexts
        for i in range(10):
            context = Context(weights=[float(i)])
            store.save_context(i, context)
        
        first_page = store.list_slots(limit=5, offset=0)
        second_page = store.list_slots(limit=5, offset=5)
        
        assert len(first_page) == 5
        assert len(second_page) == 5
        assert first_page[0]['slot_id'] != second_page[0]['slot_id']


class TestDeleteOperations:
    """Tests for deletion operations."""
    
    def test_delete_context(self, store):
        """Test deleting a context."""
        context = Context(weights=[1.0, 2.0])
        store.save_context(400, context)
        
        # Verify exists
        assert store.exists(400)
        assert store.load_context(400) is not None
        
        # Delete
        result = store.delete_context(400)
        assert result is True
        
        # Verify deleted
        assert not store.exists(400)
        assert store.load_context(400) is None
    
    def test_delete_nonexistent_context(self, store):
        """Test deleting a context that doesn't exist."""
        result = store.delete_context(999)
        assert result is False


class TestStats:
    """Tests for storage statistics."""
    
    def test_get_stats(self, store):
        """Test getting storage statistics."""
        # Add some contexts
        for i in range(3):
            context = Context(
                weights=np.random.randn(100).astype(np.float32)
            )
            store.save_context(i, context)
        
        stats = store.get_stats()
        
        assert stats['total_contexts'] == 3
        assert stats['total_size_bytes'] > 0
        assert 'by_model_type' in stats
        assert 'storage_dir' in stats
        assert 'db_path' in stats
    
    def test_empty_stats(self, temp_storage_dir):
        """Test stats for empty storage."""
        store = PersistentStore(storage_dir=temp_storage_dir)
        stats = store.get_stats()
        
        assert stats['total_contexts'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['cached_count'] == 0


class TestClearAndVacuum:
    """Tests for clear and vacuum operations."""
    
    def test_clear(self, store):
        """Test clearing all data."""
        # Add contexts
        for i in range(5):
            context = Context(weights=[float(i)])
            store.save_context(i, context)
        
        # Clear
        store.clear()
        
        # Verify cleared
        assert store.get_stats()['total_contexts'] == 0
        assert store.load_context(0) is None
    
    def test_vacuum(self, store):
        """Test vacuum operation."""
        # Add and delete some data
        for i in range(10):
            context = Context(weights=[float(i)])
            store.save_context(i, context)
        
        for i in range(5):
            store.delete_context(i)
        
        # Vacuum should not raise
        store.vacuum()


class TestExists:
    """Tests for exists method."""
    
    def test_exists_true(self, store):
        """Test exists returns True for existing slot."""
        context = Context(weights=[1.0])
        store.save_context(500, context)
        
        assert store.exists(500)
    
    def test_exists_false(self, store):
        """Test exists returns False for non-existing slot."""
        assert not store.exists(999)


class TestConcurrency:
    """Tests for thread-safety aspects."""
    
    def test_concurrent_saves(self, store):
        """Test saving from multiple threads."""
        import threading
        import time
        
        errors = []
        slots_written = []
        
        def save_context(thread_id):
            try:
                context = Context(weights=[float(thread_id)])
                store.save_context(600 + thread_id, context)
                slots_written.append(600 + thread_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=save_context, args=(i,)) for i in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(errors) == 0
        assert len(slots_written) == 5
        
        # All should be loadable
        for slot_id in slots_written:
            loaded = store.load_context(slot_id)
            assert loaded is not None
    
    def test_concurrent_reads(self, store):
        """Test reading from multiple threads."""
        import threading
        
        # Create a context
        context = Context(weights=[1.0, 2.0, 3.0])
        store.save_context(700, context)
        
        errors = []
        results = []
        
        def load_context(thread_id):
            try:
                loaded = store.load_context(700)
                results.append(loaded)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=load_context, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All reads should succeed
        assert len(errors) == 0
        assert len(results) == 10
        for loaded in results:
            assert loaded is not None


class TestAtomicWrites:
    """Tests for atomic write behavior."""
    
    def test_write_creates_weight_file(self, store):
        """Test that write creates numpy weight file."""
        context = Context(weights=np.array([1.0, 2.0, 3.0]))
        store.save_context(800, context)
        
        weight_path = store._get_weight_path(800)
        assert weight_path.exists()
    
    def test_checksum_validation(self, store):
        """Test that checksum is stored and retrievable."""
        context = Context(weights=[1.0, 2.0, 3.0])
        store.save_context(900, context)
        
        metadata = store.get_metadata(900)
        assert metadata['checksum'] is not None
        assert len(metadata['checksum']) == 32  # MD5 hex digest


class TestContextManager:
    """Tests for context manager protocol."""
    
    def test_context_manager(self, temp_storage_dir):
        """Test using store as context manager."""
        with PersistentStore(storage_dir=temp_storage_dir) as store:
            context = Context(weights=[1.0])
            store.save_context(1000, context)
        
        # Should be able to reopen and read
        with PersistentStore(storage_dir=temp_storage_dir) as store:
            loaded = store.load_context(1000)
            assert loaded is not None
            assert list(loaded.get_weights()) == [1.0]


class TestSerializationEdgeCases:
    """Tests for edge cases in serialization."""
    
    def test_empty_weights(self, store):
        """Test saving context with empty weights."""
        context = Context(weights=[])
        store.save_context(1100, context)
        
        loaded = store.load_context(1100)
        assert loaded is not None
        assert len(loaded.get_weights()) == 0
    
    def test_complex_metadata(self, store):
        """Test saving context with complex nested metadata."""
        metadata = {
            "model_type": "LLM",
            "version": "1.0.0",
            "layers": ["embedding", "attention", "ffn"],
            "config": {
                "attention_heads": 8,
                "hidden_size": 768
            }
        }
        context = Context(
            weights=[1.0],
            metadata=metadata
        )
        
        store.save_context(1200, context)
        loaded = store.load_context(1200)
        
        assert loaded.get_metadata("model_type") == "LLM"
        assert loaded.get_metadata("version") == "1.0.0"
    
    def test_empty_config_and_metadata(self, store):
        """Test saving context with empty config and metadata."""
        context = Context(weights=[1.0, 2.0])
        
        store.save_context(1300, context)
        loaded = store.load_context(1300)
        
        assert loaded.get_config() == {}
        assert loaded.get_metadata("model_type") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
