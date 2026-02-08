"""Tests for the LLMNode class."""

import pytest
import asyncio
from python.llm_node import LLMNode
from python.context import Context


class TestLLMNodeInitialization:
    """Tests for LLMNode initialization."""

    def test_default_initialization(self):
        """Test LLMNode initializes with defaults."""
        node = LLMNode(slot_index=0)
        assert node.get_slot_index() == 0
        assert node._model_path == "models/default.bin"
        assert node._parameter_count == 0

    def test_custom_model_path(self):
        """Test LLMNode with custom model path."""
        node = LLMNode(slot_index=0, model_path="custom/path.bin")
        assert node._model_path == "custom/path.bin"


class TestLLMNodeContextBuilding:
    """Tests for LLMNode context building."""

    def test_build_context_returns_context(self, llm_node_slot_0):
        """Test build_context returns a Context instance."""
        context = llm_node_slot_0.build_context()
        assert isinstance(context, Context)

    def test_build_context_sets_model_type_metadata(self, llm_node_slot_0):
        """Test build_context sets model_type to LLM."""
        context = llm_node_slot_0.build_context()
        assert context.get_metadata("model_type") == "LLM"

    def test_build_context_sets_slot_index_metadata(self, llm_node_slot_0):
        """Test build_context sets slot_index metadata."""
        context = llm_node_slot_0.build_context()
        assert context.get_metadata("slot_index") == "0"

    def test_build_context_sets_model_path_metadata(self):
        """Test build_context sets model_path metadata."""
        node = LLMNode(slot_index=0, model_path="test/path.bin")
        context = node.build_context()
        assert context.get_metadata("model_path") == "test/path.bin"

    def test_build_context_sets_config(self, llm_node_slot_0):
        """Test build_context sets required config values."""
        context = llm_node_slot_0.build_context()
        config = context.get_config()
        
        assert "temperature" in config
        assert "top_p" in config
        assert "max_tokens" in config
        assert "learning_rate" in config

    def test_build_context_generates_weights(self, llm_node_slot_0):
        """Test build_context generates weights."""
        context = llm_node_slot_0.build_context()
        weights = context.get_weights()
        
        assert len(weights) > 0
        assert all(isinstance(w, float) for w in weights)

    def test_build_context_parameter_count(self, llm_node_slot_0):
        """Test build_context sets correct parameter count."""
        context = llm_node_slot_0.build_context()
        assert llm_node_slot_0.get_parameter_count() == 100

    def test_different_slots_have_different_weights(self):
        """Test that different slots produce different weights."""
        node0 = LLMNode(slot_index=0)
        node5 = LLMNode(slot_index=5)
        
        context0 = node0.build_context()
        context5 = node5.build_context()
        
        weights0 = context0.get_weights()
        weights5 = context5.get_weights()
        
        # Different slots should have different weights
        assert weights0 != weights5

    def test_same_slot_produces_same_weights(self):
        """Test that the same slot produces the same weights."""
        node1 = LLMNode(slot_index=3)
        node2 = LLMNode(slot_index=3)
        
        context1 = node1.build_context()
        context2 = node2.build_context()
        
        # Same seed should produce same weights
        assert context1.get_weights() == context2.get_weights()

    def test_slot_0_has_100_params(self):
        """Test that slot 0 has exactly 100 parameters."""
        node = LLMNode(slot_index=0)
        node.build_context()
        assert node.get_parameter_count() == 100

    def test_slot_5_has_350_params(self):
        """Test that slot 5 has correct parameter count (100 + 5*50)."""
        node = LLMNode(slot_index=5)
        node.build_context()
        assert node.get_parameter_count() == 350


class TestLLMNodeAsyncContextBuilding:
    """Tests for LLMNode async context building."""

    @pytest.mark.asyncio
    async def test_async_build_context_returns_context(self, llm_node_slot_0):
        """Test async_build_context returns a Context instance."""
        context = await llm_node_slot_0.async_build_context()
        assert isinstance(context, Context)

    @pytest.mark.asyncio
    async def test_async_build_context_produces_same_result(self, llm_node_slot_0):
        """Test async_build_context produces same result as sync version."""
        sync_context = llm_node_slot_0.build_context()
        async_context = await llm_node_slot_0.async_build_context()
        
        assert sync_context.get_weights() == async_context.get_weights()
        assert sync_context.get_config() == async_context.get_config()


class TestLLMNodeModelLoading:
    """Tests for LLMNode model loading."""

    def test_load_model_updates_path(self, llm_node_slot_0):
        """Test load_model updates model path."""
        llm_node_slot_0.load_model("new/path.bin")
        assert llm_node_slot_0._model_path == "new/path.bin"

    def test_load_model_updates_parameter_count(self, llm_node_slot_0):
        """Test load_model updates parameter count."""
        llm_node_slot_0.load_model("test.bin")
        # Slot 0 should have 100 parameters
        assert llm_node_slot_0.get_parameter_count() == 100


class TestLLMNodeRepr:
    """Tests for LLMNode __repr__ method."""

    def test_repr_includes_class_name(self, llm_node_slot_0):
        """Test __repr__ includes class name."""
        result = repr(llm_node_slot_0)
        assert "LLMNode" in result

    def test_repr_includes_slot_index(self, llm_node_slot_0):
        """Test __repr__ includes slot index."""
        result = repr(llm_node_slot_0)
        assert "slot_index=0" in result

    def test_repr_includes_model_path(self):
        """Test __repr__ includes model path."""
        node = LLMNode(slot_index=0, model_path="test.bin")
        result = repr(node)
        assert "test.bin" in result

    def test_repr_includes_parameter_count(self, llm_node_slot_0):
        """Test __repr__ includes parameter count."""
        llm_node_slot_0.build_context()
        result = repr(llm_node_slot_0)
        assert "parameters=100" in result
