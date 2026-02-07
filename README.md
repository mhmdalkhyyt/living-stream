# Living Stream - Central Node System

A high-performance C++ application for managing AI model contexts (LLMs, CNNs) with indexed slots and cached context building.

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           Central Node System               │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │        Context<T> (Built)           │◄───┼── Cacheable, immutable after build
│  │  - model weights (copied)           │    │
│  │  - config params (copied)           │    │
│  │  - metadata (copied)                │    │
│  │  - Getters/Setters interface        │    │
│  └─────────────────────────────────────┘    │
│                    ▲                        │
│                    │ build()                │
│  ┌─────────────────────────────────────┐    │
│  │      Node<T> (Abstract base)        │    │
│  │  - slot index                       │    │
│  │  - buildContext()                   │    │
│  │  - asyncBuildContext()              │    │
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
│  │  ContextCache (LRU/in-memory)       │    │
│  │  - cacheContext(id, context)        │    │
│  │  - getCachedContext(id)             │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Key Features

- **Slot-based Storage**: Models indexed in a data structure for O(1) access
- **Context Building**: Load slot → build context (deep copy)
- **Caching**: Built contexts cached for reuse
- **Async Support**: Non-blocking context building with `std::future`
- **Getter/Setter API**: Encapsulated context access
- **Type Abstraction**: `Node<T>` base class with `LLMNode`, `CNNNode` specializations

## Design Decisions

- **No PyTorch tensors** for storage (too much overhead for model registry)
- **Deep copy** context building for thread safety
- **Immutable cached contexts** with `std::shared_ptr<const Context<T>>`
- **Thread-safe cache** using `std::shared_mutex`

## Project Structure

```
living-stream/
├── README.md
├── CMakeLists.txt
├── src/
│   ├── core/
│   │   ├── Node.hpp
│   │   ├── Context.hpp
│   │   └── ContextCache.hpp
│   ├── models/
│   │   ├── LLMNode.hpp
│   │   └── CNNNode.hpp
│   └── main.cpp
├── include/
└── memory-bank/
    ├── project-vision.md
    ├── system-architecture.md
    ├── active-context.md
    ├── progress.md
    └── lessons-learned.md
```

## Memory Bank

The `memory-bank/` directory contains:
- `project-vision.md` - Core purpose and goals
- `system-architecture.md` - Detailed technical design
- `active-context.md` - Current work state
- `progress.md` - Milestones and achievements
- `lessons-learned.md` - Development insights

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Usage

See source files for API usage examples.

## License

MIT
