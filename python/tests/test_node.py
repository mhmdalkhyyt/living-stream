"""Tests for the abstract Node base class."""

import pytest
from abc import ABC
from python.node import Node
from python.context import Context


class TestNodeAbstractClass:
    """Tests for the Node abstract base class."""

    def test_node_is_abstract(self):
        """Test that Node cannot be instantiated directly."""
        # Node should be abstract (cannot be instantiated)
        assert issubclass(Node, ABC)

    def test_abstract_methods_exist(self):
        """Test that Node has required abstract methods."""
        # The abstract methods should be defined
        assert hasattr(Node, 'build_context')
        assert hasattr(Node, 'async_build_context')

    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclasses can be instantiated."""
        # This should not raise an error
        class ConcreteNode(Node):
            def __init__(self, slot_index):
                super().__init__(slot_index)
            
            def build_context(self):
                return Context()
            
            def async_build_context(self):
                return Context()
        
        node = ConcreteNode(slot_index=0)
        assert node.get_slot_index() == 0

    def test_slot_index_stored(self):
        """Test that slot index is stored correctly."""
        class ConcreteNode(Node):
            def __init__(self, slot_index):
                super().__init__(slot_index)
            
            def build_context(self):
                return Context()
            
            def async_build_context(self):
                return Context()
        
        node = ConcreteNode(slot_index=42)
        assert node.get_slot_index() == 42

    def test_repr_includes_class_name_and_slot(self):
        """Test that __repr__ includes class name and slot."""
        class ConcreteNode(Node):
            def __init__(self, slot_index):
                super().__init__(slot_index)
            
            def build_context(self):
                return Context()
            
            def async_build_context(self):
                return Context()
        
        node = ConcreteNode(slot_index=5)
        result = repr(node)
        assert "ConcreteNode" in result
        assert "slot_index=5" in result
