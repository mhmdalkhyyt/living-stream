"""Pytest configuration and shared fixtures."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.llm_node import LLMNode
from python.cnn_node import CNNNode
from python.context import Context
from python.context_cache import ContextCache


@pytest.fixture
def sample_weights():
    """Sample weights for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {"temperature": 0.7, "top_p": 0.9}


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {"model_type": "test", "version": "1.0"}


@pytest.fixture
def sample_context(sample_weights, sample_config, sample_metadata):
    """Create a sample Context instance."""
    context = Context(
        weights=sample_weights,
        config=sample_config,
        metadata=sample_metadata
    )
    return context


@pytest.fixture
def llm_node_slot_0():
    """Create an LLMNode at slot 0."""
    return LLMNode(slot_index=0)


@pytest.fixture
def llm_node_slot_5():
    """Create an LLMNode at slot 5."""
    return LLMNode(slot_index=5)


@pytest.fixture
def cnn_node_slot_0():
    """Create a CNNNode at slot 0."""
    return CNNNode(slot_index=0)


@pytest.fixture
def cnn_node_slot_2():
    """Create a CNNNode at slot 2."""
    return CNNNode(slot_index=2)


@pytest.fixture
def empty_cache():
    """Create an empty ContextCache."""
    return ContextCache(max_size=10)


@pytest.fixture
def populated_cache(llm_node_slot_0, cnn_node_slot_0):
    """Create a ContextCache with some cached contexts."""
    cache = ContextCache(max_size=10)
    
    # Build and cache contexts
    llm_context = llm_node_slot_0.build_context()
    cnn_context = cnn_node_slot_0.build_context()
    
    cache.cache_context(0, llm_context)
    cache.cache_context(1, cnn_context)
    
    return cache
