# System Architecture

## Component Overview

### Node (Abstract Base Class)
- **Purpose**: Generic interface for all model nodes
- **Key Methods**:
  - `build_context()` - Synchronous context building
  - `async_build_context()` - Async context building returning `Awaitable[Context]`
  - `get_slot_index()` - Returns the slot index
- **Subclasses**: `LLMNode`, `CNNNode`
- **Location**: `python/node.py`

### Context
- **Purpose**: Immutable dataclass containing model data
- **Internal State**:
  - Weights (list or numpy array)
  - Configuration parameters (dict)
  - Metadata (dict)
- **API**: Public getters/setters for all fields
- **Thread Safety**: Immutable after construction (copy on set)
- **Location**: `python/context.py`

### ContextCache
- **Purpose**: LRU-style cache for built contexts (in-memory)
- **Key Methods**:
  - `cache_context(id, context)` - Store a built context
  - `get_cached_context(id)` - Retrieve cached context
- **Thread Safety**: Uses `threading.RLock`
- **Implementation**: Wraps `cachetools.LRUCache`
- **Location**: `python/context_cache.py`

### PersistentStore
- **Purpose**: Persistent storage for contexts using SQLite + numpy files
- **Storage Layout**:
  - `index.db` - SQLite database for metadata, config, indexing
  - `slots/*.npy` - Memory-mappable numpy arrays for weights
  - `wal/` - write_to_file-ahead-log for atomic operations
- **Key Methods**:
  - `save_context(slot_id, context)` - Atomic write with temp file + rename
  - `load_context(slot_id)` - Memory-mapped numpy loading
  - `get_metadata(slot_id)` - Fast metadata-only query
  - `list_slots()` - List/filter available contexts
- **Thread Safety**: Thread-local connections, RLock for cache
- **Features**:
  - SQLite WAL mode for concurrent reads during writes
  - LRU cache for hot contexts (configurable size)
  - Checksum validation (MD5)
  - Model type indexing and filtering
- **Location**: `python/storage.py`

## Data Flow

### In-Memory Caching (ContextCache)
```
1. Node(slot_index) → Load model from slot
2. node.build_context() → Create Context instance
3. cache.cache_context(id, context) → Store in cache
4. cache.get_cached_context(id) → Retrieve cached context
5. context.get_weights() / context.set_weights() → Access data
```

### Persistent Storage (PersistentStore)
```
1. Node(slot_index) → Load model from slot
2. node.build_context() → Create Context instance
3. store.save_context(slot_id, context) → Atomic write (SQLite + .npy)
4. store.load_context(slot_id) → Memory-mapped read
5. Multiple readers can access same context concurrently
```

## Concurrency Model

- **Async Building**: `asyncio` for parallel context building
- **Cache Access**: `RLock` for thread-safe operations
- **Context Immutability**: Copy on write for setters
- **Persistent Storage**: 
  - SQLite WAL mode enables concurrent reads during writes
  - Thread-local connections per process
  - Atomic file operations via rename()
  - Unlimited concurrent reads via memory-mapped files

## Storage Performance

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| SQLite metadata query | 0.1-1ms | Indexed by slot_id |
| numpy.load(mmap_mode='r') | 1-10ms | OS handles page faults |
| Concurrent reads | Unlimited | OS file cache |
| Atomic write | 5-20ms | DB + numpy save |

## One-to-Many Reader Pattern

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│   Writer Node   │     │           Reader Nodes              │
│  (Single Proc)  │     │  (Multiple Procs/Threads)          │
├─────────────────┤     ├─────────────────────────────────────┤
│ Build Context   │────▶│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│ ↓               │     │  │ Reader 1│ │ Reader 2│ │ ...    │ │
│ Atomic Write    │     │  │ mmap()  │ │ mmap()  │ │ mmap() │ │
│ (db + .npy)     │────▶│  │ SQLite  │ │ SQLite  │ │ SQLite │ │
│ ↓               │     │  │ query   │ │ query   │ │ query  │ │
└─────────────────┘     └─────────────┴──────────┴──────────┘
```

## Model Specializations

### LLMNode
- Generates synthetic LLM weights
- Config: temperature, top_p, max_tokens, learning_rate
- Location: `python/llm_node.py`

### CNNNode
- Generates synthetic CNN layer weights
- Config: kernel_size, stride, padding, activation
- Location: `python/cnn_node.py`

### NodeHierarchy
- **Purpose**: Manages parent-child hierarchy and group memberships for nodes
- **Storage**: In-memory only (not persisted)
- **Key Methods**:
  - `set_parent(child_id, parent_id)` - Set parent with cycle detection
  - `get_parent(slot_id)` - Get parent slot
  - `get_children(slot_id)` - Get all children
  - `get_ancestors(slot_id)` - Get ancestry path (root to node)
  - `get_descendants(slot_id)` - Get all descendants (subtree)
  - `move_node(slot_id, new_parent)` - Move node to new parent
  - `add_to_group(slot_id, group_name)` - Add to group
  - `remove_from_group(slot_id, group_name)` - Remove from group
  - `get_groups(slot_id)` - Get all groups for node
  - `list_by_group(group_name)` - List all nodes in group
- **Thread Safety**: Uses `threading.RLock`
- **Features**:
  - Cycle detection prevents invalid parent relationships
  - Many-to-many group memberships
  - Tree statistics (depth, root count, etc.)
- **Location**: `python/node_hierarchy.py`

## Node Hierarchy & Groups

### Parent-Child Hierarchy
```
        [Root 1]                    [Root 2]
          /   \                         |
     [Child 1] [Child 2]           [Child 3]
          |
     [Grandchild 1]
```
- Each node has at most one parent
- Multiple children per parent allowed
- Cycle detection prevents A->B->A patterns

### Group Membership
```
Group "production": {slot 1, slot 3, slot 5}
Group "experiments": {slot 2, slot 4}
Group "llms":       {slot 1, slot 2}
```
- Nodes can belong to multiple groups
- Groups can contain multiple nodes
- Many-to-many relationship

### API Endpoints

**Hierarchy Operations:**
- `POST /nodes/{slot}/parent` - Set parent (null for root)
- `GET /nodes/{slot}/parent` - Get parent
- `GET /nodes/{slot}/children` - Get children
- `GET /nodes/{slot}/ancestors` - Get ancestry path
- `GET /nodes/{slot}/descendants` - Get subtree
- `GET /nodes/{slot}/root` - Get root of tree
- `GET /nodes/{slot}/hierarchy` - Full hierarchy info
- `GET /hierarchy/stats` - Hierarchy statistics

**Group Operations:**
- `POST /nodes/{slot}/groups` - Add to group
- `DELETE /nodes/{slot}/groups/{name}` - Remove from group
- `GET /nodes/{slot}/groups` - Get node's groups
- `GET /groups` - List all groups
- `GET /groups/{name}` - List group members
