"""Tests for the NodeHierarchy class."""

import pytest
from python.node_hierarchy import NodeHierarchy


class TestNodeHierarchyInitialization:
    """Tests for NodeHierarchy initialization."""

    def test_default_initialization(self):
        """Test hierarchy initializes empty."""
        hierarchy = NodeHierarchy()
        assert hierarchy.get_stats()['total_nodes'] == 0
        assert hierarchy.get_stats()['groups_count'] == 0

    def test_repr(self):
        """Test string representation."""
        hierarchy = NodeHierarchy()
        repr_str = repr(hierarchy)
        assert "NodeHierarchy" in repr_str


class TestParentChildHierarchy:
    """Tests for parent-child hierarchy operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = NodeHierarchy()
        # Register some nodes
        for slot_id in [1, 2, 3, 4, 5, 6]:
            self.hierarchy.register_node(slot_id)

    def test_set_parent(self):
        """Test setting parent relationship."""
        assert self.hierarchy.set_parent(2, 1) is True
        assert self.hierarchy.get_parent(2) == 1

    def test_set_parent_to_none(self):
        """Test setting node as root."""
        self.hierarchy.set_parent(2, 1)
        assert self.hierarchy.set_parent(2, None) is True
        assert self.hierarchy.get_parent(2) is None

    def test_get_children(self):
        """Test getting children."""
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 1)
        children = self.hierarchy.get_children(1)
        assert 2 in children
        assert 3 in children

    def test_get_ancestors(self):
        """Test getting ancestors."""
        self.hierarchy.set_parent(3, 2)
        self.hierarchy.set_parent(2, 1)
        ancestors = self.hierarchy.get_ancestors(3)
        assert ancestors == [1, 2]

    def test_get_descendants(self):
        """Test getting descendants."""
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 1)
        self.hierarchy.set_parent(4, 2)
        self.hierarchy.set_parent(5, 2)
        descendants = self.hierarchy.get_descendants(1)
        assert 2 in descendants
        assert 3 in descendants
        assert 4 in descendants
        assert 5 in descendants

    def test_move_node(self):
        """Test moving node to new parent."""
        self.hierarchy.set_parent(2, 1)
        assert self.hierarchy.get_parent(2) == 1
        
        assert self.hierarchy.move_node(2, 3) is True
        assert self.hierarchy.get_parent(2) == 3

    def test_cycle_detection_prevents_self_parent(self):
        """Test cycle detection prevents node being its own parent."""
        assert self.hierarchy.set_parent(1, 1) is False

    def test_cycle_detection_prevents_ancestor_parent(self):
        """Test cycle detection prevents node being child of its descendant."""
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 2)
        # Trying to make 1 a child of 3 would create cycle
        assert self.hierarchy.set_parent(1, 3) is False

    def test_is_ancestor_of(self):
        """Test ancestor checking."""
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 2)
        assert self.hierarchy.is_ancestor_of(1, 3) is True
        assert self.hierarchy.is_ancestor_of(1, 2) is True
        assert self.hierarchy.is_ancestor_of(2, 3) is True
        assert self.hierarchy.is_ancestor_of(3, 1) is False

    def test_is_descendant_of(self):
        """Test descendant checking."""
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 2)
        assert self.hierarchy.is_descendant_of(3, 1) is True
        assert self.hierarchy.is_descendant_of(2, 1) is True
        assert self.hierarchy.is_descendant_of(3, 2) is True
        assert self.hierarchy.is_descendant_of(1, 3) is False

    def test_get_root(self):
        """Test getting root node."""
        self.hierarchy.set_parent(3, 2)
        self.hierarchy.set_parent(2, 1)
        assert self.hierarchy.get_root(3) == 1

    def test_get_root_of_root(self):
        """Test getting root of root node."""
        self.hierarchy.set_parent(1, None)
        assert self.hierarchy.get_root(1) == 1


class TestGroupMembership:
    """Tests for group membership operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = NodeHierarchy()
        for slot_id in [1, 2, 3, 4, 5]:
            self.hierarchy.register_node(slot_id)

    def test_add_to_group(self):
        """Test adding node to group."""
        assert self.hierarchy.add_to_group(1, "production") is True
        assert self.hierarchy.is_in_group(1, "production") is True

    def test_add_to_group_duplicate_fails(self):
        """Test adding node to same group twice fails."""
        self.hierarchy.add_to_group(1, "production")
        assert self.hierarchy.add_to_group(1, "production") is False

    def test_remove_from_group(self):
        """Test removing node from group."""
        self.hierarchy.add_to_group(1, "production")
        assert self.hierarchy.remove_from_group(1, "production") is True
        assert self.hierarchy.is_in_group(1, "production") is False

    def test_remove_from_group_not_member_fails(self):
        """Test removing from group when not a member."""
        assert self.hierarchy.remove_from_group(1, "production") is False

    def test_get_groups(self):
        """Test getting groups for a node."""
        self.hierarchy.add_to_group(1, "production")
        self.hierarchy.add_to_group(1, "experiments")
        groups = self.hierarchy.get_groups(1)
        assert "production" in groups
        assert "experiments" in groups

    def test_list_by_group(self):
        """Test listing nodes in a group."""
        self.hierarchy.add_to_group(1, "production")
        self.hierarchy.add_to_group(2, "production")
        self.hierarchy.add_to_group(3, "experiments")
        
        production_members = self.hierarchy.list_by_group("production")
        assert 1 in production_members
        assert 2 in production_members
        assert 3 not in production_members

    def test_is_in_group(self):
        """Test group membership check."""
        self.hierarchy.add_to_group(1, "production")
        assert self.hierarchy.is_in_group(1, "production") is True
        assert self.hierarchy.is_in_group(1, "experiments") is False

    def test_get_all_groups(self):
        """Test getting all group names."""
        self.hierarchy.add_to_group(1, "production")
        self.hierarchy.add_to_group(2, "production")
        self.hierarchy.add_to_group(3, "experiments")
        all_groups = self.hierarchy.get_all_groups()
        assert "production" in all_groups
        assert "experiments" in all_groups


class TestNodeRegistration:
    """Tests for node registration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = NodeHierarchy()

    def test_register_node(self):
        """Test registering a node."""
        self.hierarchy.register_node(1)
        assert self.hierarchy.get_parent(1) is None
        assert self.hierarchy.get_children(1) == []

    def test_unregister_node(self):
        """Test unregistering a node."""
        self.hierarchy.register_node(1)
        self.hierarchy.register_node(2)
        self.hierarchy.set_parent(2, 1)
        
        self.hierarchy.unregister_node(2)
        assert self.hierarchy.get_parent(2) is None

    def test_unregister_node_with_children(self):
        """Test unregistering node also removes descendants."""
        self.hierarchy.register_node(1)
        self.hierarchy.register_node(2)
        self.hierarchy.register_node(3)
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 2)
        
        self.hierarchy.unregister_node(1)
        # Node 3 should no longer be registered either
        assert self.hierarchy.get_parent(3) is None


class TestStatistics:
    """Tests for hierarchy statistics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = NodeHierarchy()

    def test_get_stats_empty(self):
        """Test stats on empty hierarchy."""
        stats = self.hierarchy.get_stats()
        assert stats['total_nodes'] == 0
        assert stats['root_nodes'] == 0
        assert stats['trees_count'] == 0
        assert stats['groups_count'] == 0

    def test_get_stats_with_hierarchy(self):
        """Test stats with hierarchy."""
        for slot_id in [1, 2, 3, 4, 5]:
            self.hierarchy.register_node(slot_id)
        
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.set_parent(3, 1)
        self.hierarchy.set_parent(4, 2)
        
        stats = self.hierarchy.get_stats()
        assert stats['total_nodes'] == 5
        assert stats['root_nodes'] == 2  # 1 and 5 are roots
        assert stats['trees_count'] == 2

    def test_get_stats_with_groups(self):
        """Test stats with groups."""
        self.hierarchy.register_node(1)
        self.hierarchy.add_to_group(1, "production")
        self.hierarchy.add_to_group(1, "experiments")
        
        stats = self.hierarchy.get_stats()
        assert stats['groups_count'] == 2


class TestClear:
    """Tests for clearing hierarchy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hierarchy = NodeHierarchy()
        for slot_id in [1, 2, 3]:
            self.hierarchy.register_node(slot_id)
        self.hierarchy.set_parent(2, 1)
        self.hierarchy.add_to_group(1, "production")

    def test_clear(self):
        """Test clearing hierarchy."""
        self.hierarchy.clear()
        stats = self.hierarchy.get_stats()
        assert stats['total_nodes'] == 0
        assert stats['groups_count'] == 0


class TestComplexScenarios:
    """Tests for complex hierarchy scenarios."""

    def test_complex_tree(self):
        """Test complex tree structure."""
        hierarchy = NodeHierarchy()
        
        # Create tree: 1 -> 2, 3; 2 -> 4, 5; 3 -> 6
        for slot_id in range(1, 7):
            hierarchy.register_node(slot_id)
        
        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 1)
        hierarchy.set_parent(4, 2)
        hierarchy.set_parent(5, 2)
        hierarchy.set_parent(6, 3)
        
        # Check children
        assert hierarchy.get_children(1) == [2, 3]
        assert hierarchy.get_children(2) == [4, 5]
        assert hierarchy.get_children(3) == [6]
        
        # Check descendants
        descendants_1 = hierarchy.get_descendants(1)
        assert set(descendants_1) == {2, 3, 4, 5, 6}
        
        descendants_2 = hierarchy.get_descendants(2)
        assert set(descendants_2) == {4, 5}
        
        # Check ancestors
        assert hierarchy.get_ancestors(4) == [1, 2]
        assert hierarchy.get_ancestors(6) == [1, 3]

    def test_multiple_groups_per_node(self):
        """Test node belonging to multiple groups."""
        hierarchy = NodeHierarchy()
        hierarchy.register_node(1)
        
        hierarchy.add_to_group(1, "production")
        hierarchy.add_to_group(1, "critical")
        hierarchy.add_to_group(1, "llm-models")
        
        groups = hierarchy.get_groups(1)
        assert len(groups) == 3
        assert "production" in groups
        assert "critical" in groups
        assert "llm-models" in groups

    def test_multiple_nodes_per_group(self):
        """Test group containing multiple nodes."""
        hierarchy = NodeHierarchy()
        
        for slot_id in range(1, 6):
            hierarchy.register_node(slot_id)
            hierarchy.add_to_group(slot_id, "production")
        
        members = hierarchy.list_by_group("production")
        assert len(members) == 5
        assert set(members) == {1, 2, 3, 4, 5}

    def test_group_cleanup_on_empty(self):
        """Test group is removed when last member leaves."""
        hierarchy = NodeHierarchy()
        hierarchy.register_node(1)
        hierarchy.add_to_group(1, "solo-group")
        
        assert "solo-group" in hierarchy.get_all_groups()
        
        hierarchy.remove_from_group(1, "solo-group")
        
        assert "solo-group" not in hierarchy.get_all_groups()
