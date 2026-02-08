# Node Hierarchy & Groups Documentation

## Overview

Added in Phase 6, this feature enables context nodes to have hierarchical relationships and group associations. Nodes can be organized into tree structures (parent-child) and grouped by categories.

## Architecture

### NodeHierarchy Class

**Location:** `python/node_hierarchy.py`

**Purpose:** Manages parent-child hierarchy and group memberships for nodes in-memory.

**Key Data Structures:**
- `_parent_map`: Dict[int, Optional[int]] - Maps slot_id -> parent_slot_id (None for root)
- `_children_map`: Dict[int, Set[int]] - Maps parent -> set of children
- `_group_members`: Dict[str, Set[int]] - Maps group_name -> set of slots
- `_slot_groups`: Dict[int, Set[str]] - Maps slot -> set of groups

### Thread Safety
- All operations use `RLock` for thread-safe access
- Safe for concurrent access from multiple threads

---

## Parent-Child Hierarchy

### Tree Structure

```
        [Root 1]                    [Root 2]
          /   \                         |
     [Child 1] [Child 2]           [Child 3]
          |
     [Grandchild 1]
```

### Key Operations

| Method | Description |
|--------|-------------|
| `set_parent(child_id, parent_id)` | Set parent with cycle detection |
| `get_parent(slot_id)` | Get parent slot (None = root) |
| `get_children(slot_id)` | Get all direct children |
| `get_ancestors(slot_id)` | Get path from root to node |
| `get_descendants(slot_id)` | Get all nodes in subtree |
| `move_node(slot_id, new_parent)` | Move to new parent |
| `is_ancestor_of(a, d)` | Check if a is ancestor of d |
| `is_descendant_of(d, a)` | Check if d is descendant of a |
| `get_root(slot_id)` | Get root of hierarchy tree |

### Cycle Detection
The hierarchy prevents invalid relationships that would create cycles:
- A node cannot be its own ancestor
- A node cannot be a descendant of itself
- `set_parent()` returns `False` if cycle detected

### Example Usage

```python
from python.node_hierarchy import NodeHierarchy

hierarchy = NodeHierarchy()

# Register nodes
for slot in [1, 2, 3, 4, 5, 6]:
    hierarchy.register_node(slot)

# Build tree: 1 -> 2, 3; 2 -> 4, 5; 3 -> 6
hierarchy.set_parent(2, 1)
hierarchy.set_parent(3, 1)
hierarchy.set_parent(4, 2)
hierarchy.set_parent(5, 2)
hierarchy.set_parent(6, 3)

# Query
print(hierarchy.get_children(1))      # [2, 3]
print(hierarchy.get_ancestors(4))      # [1, 2]
print(hierarchy.get_descendants(1))    # [2, 3, 4, 5, 6]
print(hierarchy.get_root(4))           # 1
```

---

## Group Membership

### Many-to-Many Relationships
- A node can belong to multiple groups
- A group can contain multiple nodes

### Key Operations

| Method | Description |
|--------|-------------|
| `add_to_group(slot_id, group_name)` | Add to group |
| `remove_from_group(slot_id, group_name)` | Remove from group |
| `get_groups(slot_id)` | Get all groups for node |
| `list_by_group(group_name)` | Get all nodes in group |
| `is_in_group(slot_id, group_name)` | Check membership |
| `get_all_groups()` | List all group names |

### Example Usage

```python
hierarchy = NodeHierarchy()
hierarchy.register_node(1)

# Add to groups
hierarchy.add_to_group(1, "production")
hierarchy.add_to_group(1, "critical")
hierarchy.add_to_group(1, "llm-models")

# Query
print(hierarchy.get_groups(1))              # ["critical", "llm-models", "production"]
print(hierarchy.list_by_group("production")) # [1]
print(hierarchy.get_all_groups())            # ["critical", "llm-models", "production"]
```

---

## REST API Endpoints

### Hierarchy Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/nodes/{slot}/parent` | Set parent |
| GET | `/nodes/{slot}/parent` | Get parent |
| GET | `/nodes/{slot}/children` | Get children |
| GET | `/nodes/{slot}/ancestors` | Get ancestry path |
| GET | `/nodes/{slot}/descendants` | Get subtree |
| GET | `/nodes/{slot}/root` | Get root |
| GET | `/nodes/{slot}/hierarchy` | Full hierarchy info |
| GET | `/hierarchy/stats` | Statistics |

### Group Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/nodes/{slot}/groups` | Add to group |
| DELETE | `/nodes/{slot}/groups/{name}` | Remove from group |
| GET | `/nodes/{slot}/groups` | Get node's groups |
| GET | `/groups` | List all groups |
| GET | `/groups/{name}` | List group members |

### Request/Response Models

**ParentRequest:**
```json
{
  "parent_slot": 1  // null for root
}
```

**GroupRequest:**
```json
{
  "group_name": "production"
}
```

**HierarchyResponse:**
```json
{
  "slot": 1,
  "parent": 0,
  "children": [2, 3],
  "ancestors": [0],
  "groups": ["production"]
}
```

**HierarchyStatsResponse:**
```json
{
  "total_nodes": 10,
  "root_nodes": 3,
  "trees_count": 3,
  "groups_count": 5,
  "avg_tree_depth": 2.5,
  "max_tree_depth": 4
}
```

---

## Integration with Node Class

The `Node` base class (`python/node.py`) now includes hierarchy support:

```python
class Node(ABC):
    def __init__(self, slot_index: int):
        self._slot_index = slot_index
        self._parent_slot: Optional[int] = None
        self._groups: Set[str] = set()
    
    # Parent methods
    def set_parent(self, parent_slot: Optional[int]) -> None
    def get_parent(self) -> Optional[int]
    
    # Group methods
    def add_to_group(self, group_name: str) -> bool
    def remove_from_group(self, group_name: str) -> bool
    def get_groups(self) -> Set[str]
    def is_in_group(self, group_name: str) -> bool
```

---

## Global State Management

The API server maintains global hierarchy state:

```python
# Global variables in api_server.py
_nodes: Dict[int, Union[LLMNode, CNNNode]] = {}
_cache: ContextCache = ContextCache()
_hierarchy: NodeHierarchy = NodeHierarchy()
```

The lifespan handler ensures clean state between runs:
- Clears `_nodes`, `_cache`, and `_hierarchy` on startup/shutdown

---

## Statistics

Get hierarchy statistics:
```python
stats = hierarchy.get_stats()
# {
#     'total_nodes': 10,
#     'root_nodes': 3,
#     'trees_count': 3,
#     'groups_count': 5,
#     'avg_tree_depth': 2.5,
#     'max_tree_depth': 4
# }
```

---

## Testing

All tests are in `python/tests/test_node_hierarchy.py`:
- 33 comprehensive tests covering:
  - Initialization
  - Parent-child operations
  - Group membership
  - Node registration
  - Statistics
  - Complex scenarios

Run tests:
```bash
cd python
python3 -m pytest tests/test_node_hierarchy.py -v
```

---

## Limitations

1. **In-memory only**: Hierarchy data is not persisted
   - Data is lost when server restarts
   - Consider adding persistence if needed

2. **No circular references**: Cycle detection prevents invalid hierarchies

3. **Single-parent**: Each node has at most one parent
   - Tree structure, not DAG

4. **No cross-tree operations**: Operations are per-tree

---

## Future Enhancements

Potential improvements:
1. Persist hierarchy to SQLite
2. Support for node templates
3. Hierarchy-aware context aggregation
4. Batch operations for groups
5. Hierarchy versioning/branching
