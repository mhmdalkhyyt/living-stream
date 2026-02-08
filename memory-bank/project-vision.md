# Project Vision

**Living Stream** - A Python-based central node system for managing AI model contexts.

## Core Purpose

Create a system that:
- Stores AI models (LLMs, CNNs) in an indexed slot data structure
- Builds context from model slots on demand
- Caches built contexts for fast retrieval
- Supports async context building via asyncio
- Provides clean getter/setter API for context access

## Goals

1. **Simplicity First**: Clean Python code over complex C++ patterns
2. **Type Abstraction**: Abstract `Node` base class with specialized implementations
3. **Thread Safety**: Safe concurrent access using threading.RLock
4. **Standard Library**: Use standard libraries and well-maintained packages (cachetools, numpy)

## Anti-Goals

- No heavy ML frameworks (PyTorch, TensorFlow) for storage
- No complex async patterns (keep it simple with asyncio)
- No complex dependency chains
- No monolithic single-file implementations
