# System Architecture

## Component Overview

### Node<T> (Abstract Base Class)
- **Purpose**: Generic interface for all model nodes
- **Key Methods**:
  - `buildContext()` - Synchronous context building
  - `asyncBuildContext()` - Asynchronous context building returning `std::future`
  - `getSlotIndex()` - Returns the slot index
- **Subclasses**: `LLMNode`, `CNNNode`

### Context<T>
- **Purpose**: Immutable, built context containing model data
- **Internal State**:
  - Model weights (copied)
  - Configuration parameters (copied)
  - Metadata (copied)
- **API**: Public getters/setters for all fields
- **Thread Safety**: Immutable after construction

### ContextCache
- **Purpose**: LRU-style cache for built contexts
- **Key Methods**:
  - `cacheContext(id, context)` - Store a built context
  - `getCachedContext(id)` - Retrieve cached context
- **Thread Safety**: Uses `std::shared_mutex`

## Data Flow

```
1. Node(slot_index) → Load model from slot
2. node.buildContext() → Create deep copy Context<T>
3. cache.cacheContext(id, context) → Store in cache
4. cache.getCachedContext(id) → Retrieve cached context
5. context.getWeights() / context.setWeights() → Access data
```

## Concurrency Model

- **Async Building**: `std::async` or thread pool for parallel context building
- **Cache Access**: Readers share lock, writers exclusive lock
- **Context Immutability**: Once built, context is const (except via setters)
