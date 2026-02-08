"""Tests for the REST API Server."""

import pytest
from fastapi.testclient import TestClient

from python.api_server import app, _nodes, _cache, _hierarchy


@pytest.fixture
def client():
    """Create a test client with fresh state."""
    # Clear global state before each test
    _nodes.clear()
    _cache.clear()
    _hierarchy.clear()
    return TestClient(app)


@pytest.fixture
def sample_node():
    """Sample node creation payload."""
    return {
        "slot": 0,
        "node_type": "llm",
        "model_path": "models/test.bin"
    }


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "living-stream-api"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "living-stream-api"


class TestStatusEndpoint:
    """Tests for status endpoint."""
    
    def test_status_empty(self, client):
        """Test status when no nodes exist."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes_count"] == 0
        assert data["cached_contexts"] == 0
        assert data["cache_max_size"] == 100
        assert data["thread_pool"] == "asyncio"
        assert data["cache_policy"] == "LRU"


class TestNodeManagement:
    """Tests for node management endpoints."""
    
    def test_create_node(self, client, sample_node):
        """Test creating a new node."""
        response = client.post("/nodes", json=sample_node)
        assert response.status_code == 201
        data = response.json()
        assert data["slot"] == 0
        assert data["node_type"] == "llm"
        assert data["model_path"] == "models/test.bin"
        # parameter_count depends on model loading (0 if not loaded)
        assert "parameter_count" in data
    
    def test_create_cnn_node(self, client):
        """Test creating a CNN node."""
        response = client.post("/nodes", json={
            "slot": 1,
            "node_type": "cnn",
            "model_path": "models/cnn.bin"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["slot"] == 1
        assert data["node_type"] == "cnn"
        # layer_count depends on model loading (0 if not loaded)
        assert "layer_count" in data
    
    def test_create_node_conflict(self, client, sample_node):
        """Test creating node at existing slot fails."""
        # Create first node
        client.post("/nodes", json=sample_node)
        
        # Try to create another at same slot
        response = client.post("/nodes", json=sample_node)
        assert response.status_code == 409
    
    def test_get_node(self, client, sample_node):
        """Test getting a node by slot."""
        # Create node first
        client.post("/nodes", json=sample_node)
        
        # Get the node
        response = client.get("/nodes/0")
        assert response.status_code == 200
        data = response.json()
        assert data["slot"] == 0
        assert data["node_type"] == "llm"
    
    def test_get_node_not_found(self, client):
        """Test getting non-existent node returns 404."""
        response = client.get("/nodes/999")
        assert response.status_code == 404
    
    def test_list_nodes(self, client, sample_node):
        """Test listing all nodes."""
        # Create multiple nodes
        client.post("/nodes", json=sample_node)
        client.post("/nodes", json={
            "slot": 1,
            "node_type": "cnn",
            "model_path": "models/test.bin"
        })
        
        response = client.get("/nodes")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["slot"] == 0
        assert data[1]["slot"] == 1
    
    def test_list_nodes_empty(self, client):
        """Test listing nodes when none exist."""
        response = client.get("/nodes")
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    def test_delete_node(self, client, sample_node):
        """Test deleting a node."""
        # Create node
        client.post("/nodes", json=sample_node)
        
        # Delete it
        response = client.delete("/nodes/0")
        assert response.status_code == 200
        data = response.json()
        assert "deleted" in data["message"]
        
        # Verify it's gone
        response = client.get("/nodes/0")
        assert response.status_code == 404
    
    def test_delete_node_not_found(self, client):
        """Test deleting non-existent node returns 404."""
        response = client.delete("/nodes/999")
        assert response.status_code == 404


class TestContextBuilding:
    """Tests for context building endpoints."""
    
    def test_build_context_sync(self, client):
        """Test synchronous context building."""
        # Create node
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        
        # Build context
        response = client.post("/nodes/0/build")
        assert response.status_code == 200
        data = response.json()
        assert data["slot"] == 0
        assert "weights_count" in data["context"]
        assert data["context"]["model_type"] == "LLM"
        assert data["context"]["model_path"] == "models/test.bin"
        assert "config" in data["context"]
        assert "temperature" in data["context"]["config"]
    
    def test_build_context_sync_not_found(self, client):
        """Test building context for non-existent node."""
        response = client.post("/nodes/999/build")
        assert response.status_code == 404
    
    def test_build_context_async(self, client):
        """Test asynchronous context building."""
        # Create node
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        
        # Build context async
        response = client.post("/nodes/0/build-async")
        assert response.status_code == 200
        data = response.json()
        assert data["slot"] == 0
        assert data["context"]["model_type"] == "LLM"
    
    def test_build_cnn_context(self, client):
        """Test building context for CNN node."""
        # Create CNN node
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "cnn",
            "model_path": "models/cnn.bin"
        })
        
        # Build context
        response = client.post("/nodes/0/build")
        assert response.status_code == 200
        data = response.json()
        assert data["context"]["model_type"] == "CNN"


class TestCacheOperations:
    """Tests for cache operations endpoints."""
    
    def test_cache_context(self, client):
        """Test caching a context."""
        # Create node
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        
        # Cache context
        response = client.post("/cache/0")
        assert response.status_code == 200
        data = response.json()
        assert data["slot"] == 0
        assert data["cached"] == True
        assert data["cache_size"] == 1
    
    def test_cache_context_not_found(self, client):
        """Test caching context for non-existent node."""
        response = client.post("/cache/999")
        assert response.status_code == 404
    
    def test_get_cached_context(self, client):
        """Test retrieving cached context."""
        # Create and cache node
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        client.post("/cache/0")
        
        # Get cached context
        response = client.get("/cache/0")
        assert response.status_code == 200
        data = response.json()
        assert data["slot"] == 0
        assert data["context"]["model_type"] == "LLM"
    
    def test_get_cached_context_not_found(self, client):
        """Test getting non-existent cached context."""
        response = client.get("/cache/999")
        assert response.status_code == 404
    
    def test_get_cached_context_no_cache(self, client):
        """Test getting context that exists but isn't cached."""
        # Create node but don't cache
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        
        response = client.get("/cache/0")
        assert response.status_code == 404
    
    def test_remove_cached_context(self, client):
        """Test removing a cached context."""
        # Create and cache
        client.post("/nodes", json={
            "slot": 0,
            "node_type": "llm",
            "model_path": "models/test.bin"
        })
        client.post("/cache/0")
        
        # Remove
        response = client.delete("/cache/0")
        assert response.status_code == 200
        
        # Verify removed
        response = client.get("/cache/0")
        assert response.status_code == 404
    
    def test_remove_cached_context_not_found(self, client):
        """Test removing non-existent cached context."""
        response = client.delete("/cache/999")
        assert response.status_code == 404
    
    def test_clear_cache(self, client):
        """Test clearing all cached contexts."""
        # Create and cache multiple
        client.post("/nodes", json={"slot": 0, "node_type": "llm"})
        client.post("/nodes", json={"slot": 1, "node_type": "cnn"})
        client.post("/cache/0")
        client.post("/cache/1")
        
        # Clear cache
        response = client.delete("/cache")
        assert response.status_code == 200
        
        # Verify empty
        status_response = client.get("/status")
        assert status_response.json()["cached_contexts"] == 0
    
    def test_list_cache(self, client):
        """Test listing cached contexts."""
        # Create and cache
        client.post("/nodes", json={"slot": 0, "node_type": "llm"})
        client.post("/nodes", json={"slot": 2, "node_type": "cnn"})
        client.post("/cache/0")
        client.post("/cache/2")
        
        response = client.get("/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert 0 in data["slots"]
        assert 2 in data["slots"]
    
    def test_list_cache_empty(self, client):
        """Test listing empty cache."""
        response = client.get("/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["slots"] == []


class TestValidation:
    """Tests for request validation."""
    
    def test_invalid_slot_negative(self, client):
        """Test that negative slots are rejected."""
        response = client.post("/nodes", json={
            "slot": -1,
            "node_type": "llm"
        })
        assert response.status_code == 422  # Validation error
    
    def test_invalid_slot_too_high(self, client):
        """Test that slots above 999 are rejected."""
        response = client.post("/nodes", json={
            "slot": 1000,
            "node_type": "llm"
        })
        assert response.status_code == 422  # Validation error
    
    def test_invalid_node_type(self, client):
        """Test that invalid node types are rejected."""
        response = client.post("/nodes", json={
            "slot": 0,
            "node_type": "rnn"
        })
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test that missing required fields are rejected."""
        response = client.post("/nodes", json={})
        assert response.status_code == 422  # Validation error


class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_full_workflow(self, client):
        """Test complete workflow: create, build, cache, get."""
        # Create LLM node
        create_response = client.post("/nodes", json={
            "slot": 5,
            "node_type": "llm",
            "model_path": "models/gpt.bin"
        })
        assert create_response.status_code == 201
        
        # Build context
        build_response = client.post("/nodes/5/build")
        assert build_response.status_code == 200
        original_weights = build_response.json()["context"]["weights_count"]
        
        # Cache context
        cache_response = client.post("/cache/5")
        assert cache_response.status_code == 200
        
        # Get cached context
        get_response = client.get("/cache/5")
        assert get_response.status_code == 200
        cached_weights = get_response.json()["context"]["weights_count"]
        assert cached_weights == original_weights
        
        # Check status
        status_response = client.get("/status")
        assert status_response.json()["nodes_count"] == 1
        assert status_response.json()["cached_contexts"] == 1
        
        # Clean up
        client.delete("/nodes/5")
        client.delete("/cache")
        
        # Verify cleanup
        status_response = client.get("/status")
        assert status_response.json()["nodes_count"] == 0
        assert status_response.json()["cached_contexts"] == 0
    
    def test_multiple_nodes_workflow(self, client):
        """Test workflow with multiple nodes."""
        # Create multiple nodes
        client.post("/nodes", json={"slot": 0, "node_type": "llm"})
        client.post("/nodes", json={"slot": 1, "node_type": "cnn"})
        client.post("/nodes", json={"slot": 2, "node_type": "llm"})
        
        # Build and cache all
        client.post("/cache/0")
        client.post("/cache/1")
        client.post("/cache/2")
        
        # List nodes
        nodes_response = client.get("/nodes")
        assert len(nodes_response.json()) == 3
        
        # List cache
        cache_response = client.get("/cache")
        assert cache_response.json()["count"] == 3
        
        # Get specific cached
        cnn_response = client.get("/cache/1")
        assert cnn_response.json()["context"]["model_type"] == "CNN"
