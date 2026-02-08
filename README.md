# Living Stream - AI Model Context Manager

A Python application for managing AI model contexts (LLMs, CNNs) with indexed slots and cached context building.

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Living Stream System                │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │        Context (dataclass)          │◄───┼── Cacheable, copy-on-write
│  │  - weights (list/numpy)             │    │
│  │  - config (dict)                    │    │
│  │  - metadata (dict)                  │    │
│  │  - Getters/Setters interface        │    │
│  └─────────────────────────────────────┘    │
│                    ▲                        │
│                    │ build_context()        │
│  ┌─────────────────────────────────────┐    │
│  │      Node (Abstract Base Class)     │    │
│  │  - slot index                       │    │
│  │  - build_context()                  │    │
│  │  - async_build_context()            │    │
│  └─────────────────────────────────────┘    │
│          │                                  │
│     ┌────┴────┐                             │
│     ▼         ▼                             │
│  ┌──────┐  ┌──────┐                         │
│  │LLM   │  │ CNN  │  ... other types        │
│  │Node  │  │Node  │                         │
│  └──────┘  └──────┘                         │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  ContextCache (LRU, thread-safe)    │    │
│  │  - cachetools.LRUCache              │    │
│  │  - threading.RLock                  │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  CLI (Interactive)                  │    │
│  │  - create, build, cache, get        │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Key Features

- **Slot-based Storage**: Models indexed by slot number for O(1) access
- **Context Building**: Load slot → build context (creates new Context)
- **Caching**: Built contexts cached for reuse (LRU policy)
- **Async Support**: Non-blocking context building with `asyncio`
- **Getter/Setter API**: Encapsulated context access with copies
- **Type Abstraction**: `Node` base class with `LLMNode`, `CNNNode` specializations
- **Interactive CLI**: Full command-line interface for testing

## Design Decisions

- **Python dataclasses** for clean Context structure
- **cachetools.LRUCache** for reliable caching
- **threading.RLock** for thread-safe cache access
- **asyncio** for async operations
- **numpy** optional for efficient array operations

## Project Structure

```
living-stream/
├── README.md
├── python/
│   ├── __init__.py
│   ├── node.py           (Abstract Node class)
│   ├── context.py        (Context dataclass)
│   ├── context_cache.py  (Thread-safe LRU cache)
│   ├── llm_node.py       (LLM specialization)
│   ├── cnn_node.py       (CNN specialization)
│   ├── cli.py            (Interactive CLI)
│   └── requirements.txt
├── memory-bank/
│   ├── project-vision.md
│   ├── system-architecture.md
│   ├── active-context.md
│   ├── progress.md
│   └── lessons-learned.md
```

## Installation

```bash
cd python
pip install -r requirements.txt
```

## Usage

### Interactive CLI

```bash
cd python
python cli.py
```

Commands:
```
> create llm 0              # Create LLM at slot 0
> create cnn 1              # Create CNN at slot 1
> cache 0                   # Build and cache context
> get 0                     # Retrieve cached context
> build-async 1             # Build asynchronously
> list                      # List cached contexts
> status                    # Show system status
> help                      # Show all commands
> quit                      # Exit
```

### Programmatic Usage

```python
from llm_node import LLMNode
from cnn_node import CNNNode
from context_cache import ContextCache

# Create nodes
llm = LLMNode(slot_index=0, model_path="models/llm.bin")
cnn = CNNNode(slot_index=1, model_path="models/cnn.bin")

# Build contexts
context = llm.build_context()
print(f"Weights: {len(context.get_weights())}")

# Use cache
cache = ContextCache(max_size=100)
cache.cache_context(0, context)
cached = cache.get_cached_context(0)

# Async building
import asyncio
async def main():
    context = await llm.async_build_context()

asyncio.run(main())
```

## Dependencies

- Python 3.8+
- numpy >= 1.20.0
- cachetools >= 5.0.0

## License

MIT
