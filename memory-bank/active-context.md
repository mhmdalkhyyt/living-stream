# Active Context

## Current Phase
Project initialization and architecture documentation.

## Tasks In Progress
- [x] Define project vision
- [x] Create system architecture documentation
- [x] Initialize memory bank structure

## Next Steps
- [ ] Create CMakeLists.txt
- [ ] Implement Node<T> base class
- [ ] Implement Context<T> template
- [ ] Implement ContextCache
- [ ] Create LLMNode specialization
- [ ] Create CNNNode specialization

## Decisions Made
1. Use deep copy (not zero-copy) for context building
2. Cache built contexts with LRU semantics
3. No PyTorch tensors for storage
4. Use std::future for async operations

## Open Questions
- Cache eviction policy? (LRU, LFU, TTL?)
- Maximum cache size? (configurable or fixed?)
- Thread pool size for async operations?
