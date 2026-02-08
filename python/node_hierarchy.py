"""NodeHierarchy - Manages parent-child hierarchy and group memberships for nodes.

Provides thread-safe tree operations and group management for model nodes.
"""

from typing import Dict, List, Optional, Set
from threading import RLock


class NodeHierarchy:
    """Manages parent-child hierarchy and group memberships for nodes.
    
    Supports:
    - Tree structure with parent-child relationships
    - Many-to-many group memberships
    - Cycle detection to prevent invalid relationships
    - Thread-safe operations with RLock
    
    Attributes:
        _parent_map: Maps slot_id -> parent_slot_id (None for root nodes)
        _children_map: Maps slot_id -> Set[child_slot_id]
        _group_members: Maps group_name -> Set[slot_id]
        _slot_groups: Maps slot_id -> Set[group_name]
    """
    
    def __init__(self):
        """Initialize empty hierarchy manager."""
        self._parent_map: Dict[int, Optional[int]] = {}  # slot -> parent (None = root)
        self._children_map: Dict[int, Set[int]] = {}  # parent -> set of children
        self._group_members: Dict[str, Set[int]] = {}  # group_name -> set of slots
        self._slot_groups: Dict[int, Set[str]] = {}  # slot -> set of groups
        self._lock = RLock()
    
    # ============= Parent-Child Hierarchy Operations =============
    
    def set_parent(self, child_id: int, parent_id: Optional[int]) -> bool:
        """Set the parent of a node.
        
        Args:
            child_id: The slot ID of the child node
            parent_id: The slot ID of the parent node, or None for root
            
        Returns:
            True if successful, False if would create a cycle
        """
        with self._lock:
            # Validate parent exists
            if parent_id is not None and parent_id not in self._parent_map:
                # Auto-register the parent if not exists
                self._parent_map[parent_id] = None
                self._children_map[parent_id] = set()
            
            # Check for cycles (can't make node its own ancestor)
            if parent_id is not None and self._would_create_cycle(child_id, parent_id):
                return False
            
            # Remove from old parent
            old_parent = self._parent_map.get(child_id)
            if old_parent is not None and old_parent in self._children_map:
                self._children_map[old_parent].discard(child_id)
                if not self._children_map[old_parent]:
                    del self._children_map[old_parent]
            
            # Update parent's children set
            if parent_id is not None:
                if parent_id not in self._children_map:
                    self._children_map[parent_id] = set()
                self._children_map[parent_id].add(child_id)
            
            # Update child
            self._parent_map[child_id] = parent_id
            
            # Ensure child has children entry
            if child_id not in self._children_map:
                self._children_map[child_id] = set()
            
            return True
    
    def _would_create_cycle(self, node_id: int, potential_parent_id: int) -> bool:
        """Check if setting potential_parent_id as parent would create a cycle."""
        # Check if node_id is an ancestor of potential_parent_id
        current = potential_parent_id
        while current is not None:
            if current == node_id:
                return True
            current = self._parent_map.get(current)
        return False
    
    def get_parent(self, slot_id: int) -> Optional[int]:
        """Get the parent of a node.
        
        Args:
            slot_id: The slot ID of the node
            
        Returns:
            Parent slot ID, or None if root/not found
        """
        with self._lock:
            return self._parent_map.get(slot_id)
    
    def get_children(self, slot_id: int) -> List[int]:
        """Get all children of a node.
        
        Args:
            slot_id: The slot ID of the parent node
            
        Returns:
            List of child slot IDs
        """
        with self._lock:
            children = self._children_map.get(slot_id, set())
            return sorted(list(children))
    
    def get_ancestors(self, slot_id: int) -> List[int]:
        """Get the ancestry path from root to the node.
        
        Args:
            slot_id: The slot ID of the node
            
        Returns:
            List of ancestor slot IDs (root first, parent last)
        """
        with self._lock:
            ancestors = []
            current = self._parent_map.get(slot_id)
            while current is not None:
                ancestors.append(current)
                current = self._parent_map.get(current)
            return list(reversed(ancestors))  # root first
    
    def get_descendants(self, slot_id: int) -> List[int]:
        """Get all descendants of a node (all nodes in subtree).
        
        Args:
            slot_id: The slot ID of the root node
            
        Returns:
            List of all descendant slot IDs
        """
        with self._lock:
            descendants = []
            self._collect_descendants(slot_id, descendants)
            return sorted(descendants)
    
    def _collect_descendants(self, slot_id: int, result: List[int]) -> None:
        """Recursively collect descendants (only children, not self)."""
        children = self._children_map.get(slot_id, set())
        for child in children:
            result.append(child)
            self._collect_descendants(child, result)
    
    def move_node(self, slot_id: int, new_parent_id: Optional[int]) -> bool:
        """Move a node to a new parent.
        
        Args:
            slot_id: The slot ID of the node to move
            new_parent_id: The new parent slot ID, or None for root
            
        Returns:
            True if successful, False if would create a cycle
        """
        with self._lock:
            return self.set_parent(slot_id, new_parent_id)
    
    def is_ancestor_of(self, ancestor_id: int, descendant_id: int) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id.
        
        Args:
            ancestor_id: Potential ancestor slot ID
            descendant_id: Potential descendant slot ID
            
        Returns:
            True if ancestor_id is an ancestor of descendant_id
        """
        with self._lock:
            current = self._parent_map.get(descendant_id)
            while current is not None:
                if current == ancestor_id:
                    return True
                current = self._parent_map.get(current)
            return False
    
    def is_descendant_of(self, descendant_id: int, ancestor_id: int) -> bool:
        """Check if descendant_id is a descendant of ancestor_id.
        
        Args:
            descendant_id: Potential descendant slot ID
            ancestor_id: Potential ancestor slot ID
            
        Returns:
            True if descendant_id is a descendant of ancestor_id
        """
        with self._lock:
            return self.is_ancestor_of(ancestor_id, descendant_id)
    
    def get_root(self, slot_id: int) -> Optional[int]:
        """Get the root node of the hierarchy tree containing slot_id.
        
        Args:
            slot_id: The slot ID of any node in the tree
            
        Returns:
            Root slot ID, or None if not in hierarchy
        """
        with self._lock:
            if slot_id not in self._parent_map:
                return None
            
            ancestors = self.get_ancestors(slot_id)
            if ancestors:
                return ancestors[0]
            return slot_id  # Node is root
    
    # ============= Group Membership Operations =============
    
    def add_to_group(self, slot_id: int, group_name: str) -> bool:
        """Add a node to a group.
        
        Args:
            slot_id: The slot ID of the node
            group_name: The name of the group
            
        Returns:
            True if added, False if already in group
        """
        with self._lock:
            # Ensure group exists
            if group_name not in self._group_members:
                self._group_members[group_name] = set()
            
            # Ensure slot has group entry
            if slot_id not in self._slot_groups:
                self._slot_groups[slot_id] = set()
            
            # Add to group
            if group_name in self._slot_groups[slot_id]:
                return False  # Already in group
            
            self._group_members[group_name].add(slot_id)
            self._slot_groups[slot_id].add(group_name)
            return True
    
    def remove_from_group(self, slot_id: int, group_name: str) -> bool:
        """Remove a node from a group.
        
        Args:
            slot_id: The slot ID of the node
            group_name: The name of the group
            
        Returns:
            True if removed, False if not in group
        """
        with self._lock:
            removed = False
            
            if slot_id in self._slot_groups and group_name in self._slot_groups[slot_id]:
                self._slot_groups[slot_id].discard(group_name)
                if not self._slot_groups[slot_id]:
                    del self._slot_groups[slot_id]
                removed = True
            
            if group_name in self._group_members and slot_id in self._group_members[group_name]:
                self._group_members[group_name].discard(slot_id)
                if not self._group_members[group_name]:
                    del self._group_members[group_name]
            
            return removed
    
    def get_groups(self, slot_id: int) -> List[str]:
        """Get all groups a node belongs to.
        
        Args:
            slot_id: The slot ID of the node
            
        Returns:
            List of group names
        """
        with self._lock:
            groups = self._slot_groups.get(slot_id, set())
            return sorted(list(groups))
    
    def list_by_group(self, group_name: str) -> List[int]:
        """List all nodes in a group.
        
        Args:
            group_name: The name of the group
            
        Returns:
            List of slot IDs in the group
        """
        with self._lock:
            members = self._group_members.get(group_name, set())
            return sorted(list(members))
    
    def is_in_group(self, slot_id: int, group_name: str) -> bool:
        """Check if a node is in a group.
        
        Args:
            slot_id: The slot ID of the node
            group_name: The name of the group
            
        Returns:
            True if node is in the group
        """
        with self._lock:
            return (
                slot_id in self._slot_groups and
                group_name in self._slot_groups[slot_id]
            )
    
    def get_all_groups(self) -> List[str]:
        """Get all group names.
        
        Returns:
            List of all group names
        """
        with self._lock:
            return sorted(list(self._group_members.keys()))
    
    # ============= Node Registration =============
    
    def register_node(self, slot_id: int) -> None:
        """Register a node in the hierarchy (without parent).
        
        Args:
            slot_id: The slot ID of the node to register
        """
        with self._lock:
            if slot_id not in self._parent_map:
                self._parent_map[slot_id] = None
                self._children_map[slot_id] = set()
    
    def unregister_node(self, slot_id: int) -> None:
        """Unregister a node from the hierarchy.
        
        Removes the node and all its descendants from the hierarchy.
        
        Args:
            slot_id: The slot ID of the node to unregister
        """
        with self._lock:
            # Unregister all descendants first
            descendants = self.get_descendants(slot_id)
            for desc in descendants:
                self._unregister_single(desc)
            
            # Unregister the node itself
            self._unregister_single(slot_id)
    
    def _unregister_single(self, slot_id: int) -> None:
        """Unregister a single node (must hold lock)."""
        # Remove from parent
        parent = self._parent_map.get(slot_id)
        if parent is not None and parent in self._children_map:
            self._children_map[parent].discard(slot_id)
        
        # Remove from children
        if slot_id in self._children_map:
            del self._children_map[slot_id]
        
        # Remove from parent map
        self._parent_map.pop(slot_id, None)
        
        # Remove from groups
        groups = self._slot_groups.pop(slot_id, set())
        for group in groups:
            self._group_members[group].discard(slot_id)
            if not self._group_members[group]:
                del self._group_members[group]
    
    # ============= Statistics and Query =============
    
    def get_stats(self) -> Dict:
        """Get hierarchy statistics.
        
        Returns:
            Dict with hierarchy stats
        """
        with self._lock:
            # Count root nodes
            root_nodes = sum(1 for p in self._parent_map.values() if p is None)
            
            # Count nodes in hierarchy trees
            nodes_in_trees = set()
            for slot_id in self._parent_map:
                root = self.get_root(slot_id)
                if root is not None:
                    nodes_in_trees.add(root)
            
            # Calculate tree depths
            depths = []
            for root in nodes_in_trees:
                max_depth = self._get_max_depth(root)
                depths.append(max_depth)
            
            return {
                'total_nodes': len(self._parent_map),
                'root_nodes': root_nodes,
                'trees_count': len(nodes_in_trees),
                'groups_count': len(self._group_members),
                'avg_tree_depth': sum(depths) / len(depths) if depths else 0,
                'max_tree_depth': max(depths) if depths else 0
            }
    
    def _get_max_depth(self, slot_id: int, current_depth: int = 0) -> int:
        """Get maximum depth of subtree (must hold lock)."""
        children = self._children_map.get(slot_id, set())
        if not children:
            return current_depth
        return max(
            self._get_max_depth(child, current_depth + 1)
            for child in children
        )
    
    def clear(self) -> None:
        """Clear all hierarchy data."""
        with self._lock:
            self._parent_map.clear()
            self._children_map.clear()
            self._group_members.clear()
            self._slot_groups.clear()
    
    def __repr__(self) -> str:
        with self._lock:
            stats = self.get_stats()
            return (
                f"NodeHierarchy("
                f"nodes={stats['total_nodes']}, "
                f"roots={stats['root_nodes']}, "
                f"groups={stats['groups_count']}"
                f")"
            )
