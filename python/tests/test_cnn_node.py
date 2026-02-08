"""Tests for the CNNNode class."""

import pytest
import asyncio
from python.cnn_node import CNNNode
from python.context import Context


class TestCNNNodeInitialization:
    """Tests for CNNNode initialization."""

    def test_default_initialization(self):
        """Test CNNNode initializes with defaults."""
        node = CNNNode(slot_index=0)
        assert node.get_slot_index() == 0
        assert node._model_path == "models/default.bin"
        assert node._layer_count == 0

    def test_custom_model_path(self):
        """Test CNNNode with custom model path."""
        node = CNNNode(slot_index=0, model_path="custom/path.bin")
        assert node._model_path == "custom/path.bin"


class TestCNNNodeContextBuilding:
    """Tests for CNNNode context building."""

    def test_build_context_returns_context(self, cnn_node_slot_0):
        """Test build_context returns a Context instance."""
        context = cnn_node_slot_0.build_context()
        assert isinstance(context, Context)

    def test_build_context_sets_model_type_metadata(self, cnn_node_slot_0):
        """Test build_context sets model_type to CNN."""
        context = cnn_node_slot_0.build_context()
        assert context.get_metadata("model_type") == "CNN"

    def test_build_context_sets_slot_index_metadata(self, cnn_node_slot_0):
        """Test build_context sets slot_index metadata."""
        context = cnn_node_slot_0.build_context()
        assert context.get_metadata("slot_index") == "0"

    def test_build_context_sets_model_path_metadata(self):
        """Test build_context sets model_path metadata."""
        node = CNNNode(slot_index=0, model_path="test/path.bin")
        context = node.build_context()
        assert context.get_metadata("model_path") == "test/path.bin"

    def test_build_context_sets_layer_count_metadata(self, cnn_node_slot_0):
        """Test build_context sets layer_count metadata."""
        context = cnn_node_slot_0.build_context()
        assert context.get_metadata("layer_count") == "4"

    def test_build_context_sets_config(self, cnn_node_slot_0):
        """Test build_context sets required config values."""
        context = cnn_node_slot_0.build_context()
        config = context.get_config()
        
        assert "kernel_size" in config
        assert "stride" in config
        assert "padding" in config
        assert "activation" in config

    def test_build_context_generates_weights(self, cnn_node_slot_0):
        """Test build_context generates weights."""
        context = cnn_node_slot_0.build_context()
        weights = context.get_weights()
        
        assert len(weights) > 0
        assert all(isinstance(w, float) for w in weights)

    def test_build_context_layer_count(self, cnn_node_slot_0):
        """Test build_context sets correct layer count."""
        context = cnn_node_slot_0.build_context()
        assert cnn_node_slot_0.get_layer_count() == 4

    def test_different_slots_have_different_weights(self):
        """Test that different slots produce different weights."""
        node0 = CNNNode(slot_index=0)
        node2 = CNNNode(slot_index=2)
        
        context0 = node0.build_context()
        context2 = node2.build_context()
        
        weights0 = context0.get_weights()
        weights2 = context2.get_weights()
        
        # Different slots should have different weights
        assert weights0 != weights2

    def test_same_slot_produces_same_weights(self):
        """Test that the same slot produces the same weights."""
        node1 = CNNNode(slot_index=3)
        node2 = CNNNode(slot_index=3)
        
        context1 = node1.build_context()
        context2 = node2.build_context()
        
        # Same seed should produce same weights
        assert context1.get_weights() == context2.get_weights()

    def test_slot_0_has_4_layers(self):
        """Test that slot 0 has exactly 4 layers."""
        node = CNNNode(slot_index=0)
        node.build_context()
        assert node.get_layer_count() == 4

    def test_slot_2_has_6_layers(self):
        """Test that slot 2 has correct layer count (4 + 2)."""
        node = CNNNode(slot_index=2)
        node.build_context()
        assert node.get_layer_count() == 6

    def test_layer_sizes_progress_correctly(self, cnn_node_slot_0):
        """Test that layer sizes progress correctly through layers."""
        context = cnn_node_slot_0.build_context()
        layer_sizes = cnn_node_slot_0.get_layer_sizes()
        
        # Slot 0 has 4 layers:
        # - i=0: starts at 64, doubles to 128 after appending
        # - i=1: uses 128 (no doubling)
        # - i=2: uses 128, doubles to 256 after appending
        # - i=3: uses 256 (last layer, no doubling)
        assert len(layer_sizes) == 4
        assert layer_sizes[0] == 64
        assert layer_sizes[1] == 128
        assert layer_sizes[2] == 128
        assert layer_sizes[3] == 256


class TestCNNNodeAsyncContextBuilding:
    """Tests for CNNNode async context building."""

    @pytest.mark.asyncio
    async def test_async_build_context_returns_context(self, cnn_node_slot_0):
        """Test async_build_context returns a Context instance."""
        context = await cnn_node_slot_0.async_build_context()
        assert isinstance(context, Context)

    @pytest.mark.asyncio
    async def test_async_build_context_produces_same_result(self, cnn_node_slot_0):
        """Test async_build_context produces same result as sync version."""
        sync_context = cnn_node_slot_0.build_context()
        async_context = await cnn_node_slot_0.async_build_context()
        
        assert sync_context.get_weights() == async_context.get_weights()
        assert sync_context.get_config() == async_context.get_config()


class TestCNNNodeModelLoading:
    """Tests for CNNNode model loading."""

    def test_load_model_updates_path(self, cnn_node_slot_0):
        """Test load_model updates model path."""
        cnn_node_slot_0.load_model("new/path.bin")
        assert cnn_node_slot_0._model_path == "new/path.bin"

    def test_load_model_updates_layer_count(self, cnn_node_slot_0):
        """Test load_model updates layer count."""
        cnn_node_slot_0.load_model("test.bin")
        # Slot 0 should have 4 layers
        assert cnn_node_slot_0.get_layer_count() == 4


class TestCNNNodeLayerSizes:
    """Tests for CNNNode layer size functionality."""

    def test_get_layer_sizes_returns_copy(self, cnn_node_slot_0):
        """Test get_layer_sizes returns a copy."""
        cnn_node_slot_0.build_context()
        sizes1 = cnn_node_slot_0.get_layer_sizes()
        sizes1[0] = 999
        sizes2 = cnn_node_slot_0.get_layer_sizes()
        assert sizes2[0] != 999


class TestCNNNodeRepr:
    """Tests for CNNNode __repr__ method."""

    def test_repr_includes_class_name(self, cnn_node_slot_0):
        """Test __repr__ includes class name."""
        result = repr(cnn_node_slot_0)
        assert "CNNNode" in result

    def test_repr_includes_slot_index(self, cnn_node_slot_0):
        """Test __repr__ includes slot index."""
        result = repr(cnn_node_slot_0)
        assert "slot_index=0" in result

    def test_repr_includes_model_path(self):
        """Test __repr__ includes model path."""
        node = CNNNode(slot_index=0, model_path="test.bin")
        result = repr(node)
        assert "test.bin" in result

    def test_repr_includes_layer_count(self, cnn_node_slot_0):
        """Test __repr__ includes layer count."""
        cnn_node_slot_0.build_context()
        result = repr(cnn_node_slot_0)
        assert "layers=4" in result

    def test_repr_includes_layer_sizes(self, cnn_node_slot_0):
        """Test __repr__ includes layer sizes."""
        cnn_node_slot_0.build_context()
        result = repr(cnn_node_slot_0)
        assert "layer_sizes" in result
