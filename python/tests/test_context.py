"""Tests for the Context dataclass."""

import pytest
import numpy as np
from python.context import Context


class TestContextInitialization:
    """Tests for Context initialization."""

    def test_default_initialization(self):
        """Test Context initializes with empty defaults."""
        context = Context()
        assert context.weights == []
        assert context.config == {}
        assert context.metadata == {}

    def test_custom_initialization_with_lists(self):
        """Test Context initialization with custom values."""
        weights = [0.1, 0.2, 0.3]
        config = {"temperature": 0.7}
        metadata = {"key": "value"}
        
        context = Context(
            weights=weights,
            config=config,
            metadata=metadata
        )
        
        assert context.weights == weights
        assert context.config == config
        assert context.metadata == metadata

    def test_custom_initialization_with_numpy_arrays(self):
        """Test Context initialization with numpy arrays."""
        weights = np.array([0.1, 0.2, 0.3])
        config = {"temperature": 0.7}
        metadata = {"key": "value"}
        
        context = Context(
            weights=weights,
            config=config,
            metadata=metadata
        )
        
        np.testing.assert_array_equal(context.weights, weights)


class TestContextGettersSetters:
    """Tests for Context getter and setter methods."""

    def test_get_weights_returns_list(self, sample_context, sample_weights):
        """Test get_weights returns the weights."""
        result = sample_context.get_weights()
        assert result == sample_weights

    def test_set_weights(self, sample_context):
        """Test set_weights updates weights."""
        new_weights = [1.0, 2.0, 3.0]
        sample_context.set_weights(new_weights)
        assert sample_context.weights == new_weights

    def test_set_weights_with_numpy(self, sample_context):
        """Test set_weights with numpy array."""
        new_weights = np.array([1.0, 2.0, 3.0])
        sample_context.set_weights(new_weights)
        np.testing.assert_array_equal(sample_context.weights, new_weights)

    def test_get_config_returns_copy(self, sample_context, sample_config):
        """Test get_config returns a copy."""
        result = sample_context.get_config()
        assert result == sample_config
        # Modifying the returned dict should not affect the original
        result["new_key"] = "new_value"
        assert "new_key" not in sample_context.get_config()

    def test_set_config(self, sample_context):
        """Test set_config updates config."""
        new_config = {"temperature": 0.5, "top_p": 0.8}
        sample_context.set_config(new_config)
        assert sample_context.get_config() == new_config

    def test_get_metadata_existing_key(self, sample_context):
        """Test get_metadata returns value for existing key."""
        result = sample_context.get_metadata("model_type")
        assert result == "test"

    def test_get_metadata_missing_key(self, sample_context):
        """Test get_metadata returns empty string for missing key."""
        result = sample_context.get_metadata("nonexistent")
        assert result == ""

    def test_set_metadata(self, sample_context):
        """Test set_metadata updates metadata."""
        sample_context.set_metadata("new_key", "new_value")
        assert sample_context.get_metadata("new_key") == "new_value"


class TestContextStringRepresentation:
    """Tests for Context __str__ method."""

    def test_str_with_weights(self, sample_context):
        """Test string representation includes weights count."""
        result = str(sample_context)
        assert "weights=5" in result
        assert "config_keys" in result
        assert "metadata_keys" in result

    def test_str_empty_context(self):
        """Test string representation of empty context."""
        context = Context()
        result = str(context)
        assert "weights=0" in result
