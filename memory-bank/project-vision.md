# Project Vision

**Living Stream** - A high-performance C++ central node system for managing AI model contexts.

## Core Purpose

Create a system that:
- Stores AI models (LLMs, CNNs) in an indexed slot data structure
- Builds context from model slots on demand
- Caches built contexts for fast retrieval
- Supports async context building
- Provides clean getter/setter API for context access

## Goals

1. **Performance First**: Prioritize speed over convenience
2. **Type Abstraction**: Generic `Node<T>` base with specialized implementations
3. **Thread Safety**: Safe concurrent access to cached contexts
4. **No Overhead**: Avoid heavy dependencies like PyTorch for storage

## Anti-Goals

- No PyTorch tensors for model storage (too much overhead)
- No zero-copy semantics (explicit deep copy for safety)
- No complex dependency chains
