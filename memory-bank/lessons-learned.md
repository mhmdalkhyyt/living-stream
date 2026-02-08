# Lessons Learned

## Architecture Decisions

### dataclasses for Context
**Decision**: Used Python's `@dataclass` decorator for Context
**Reason**: Clean, immutable-like structure with automatic `__init__`, `__repr__`
**Impact**: Simple, readable code with field defaults

### cachetools for LRU Cache
**Decision**: Used `cachetools.LRUCache` for context caching
**Reason**: Battle-tested, thread-safe options, proper LRU eviction
**Impact**: Reliable caching with minimal code

### threading.RLock for Thread Safety
**Decision**: Used `threading.RLock` for cache thread safety
**Reason**: Python's GIL makes simple locking sufficient; RLock is reentrant
**Impact**: Safe concurrent access without complex locks

### asyncio for Async Operations
**Decision**: Used Python's native `async`/`await`
**Reason**: Standard library, familiar pattern, easy to understand
**Impact**: Non-blocking context building possible

### numpy for Array Support
**Decision**: Added numpy as optional dependency for weights
**Reason**: Efficient numerical operations when needed
**Impact**: Flexible - works with lists or numpy arrays

## Code Organization

### Directory Structure
**Pattern**: Separate core (Node, Context, Cache) from model specializations
**Benefit**: Clean separation of concerns, easy to extend

### File per Class
**Decision**: One Python file per class/module
**Benefit**: Clear organization, easy imports, testable

## What Works Well
1. Abstract base class pattern for Node hierarchy
2. Dataclass for Context provides clean API
3. cachetools handles LRU eviction reliably
4. CLI makes testing easy
5. Thread-safe cache with RLock

## What Could Improve
1. Currently synthetic weights - real model loading TBD
2. No configuration file support yet
3. Could use pydantic for validation
4. Could add type hints more extensively
