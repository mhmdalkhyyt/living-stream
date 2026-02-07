# Lessons Learned

## Architecture Decisions

### PyTorch Data Structures
**Decision**: Rejected PyTorch tensors for storage
**Reason**: Too much overhead (autograd, device management, gradient tracking)
**Impact**: Faster, lighter implementation with custom lightweight structures

### Zero-Copy vs Deep Copy
**Decision**: Chose deep copy context building
**Reason**: User preference for cached, immutable contexts with getter/setter API
**Impact**: Memory trade-off for thread safety and clean API

### Async Support
**Decision**: Use `std::future` for async context building
**Reason**: Standard library support, no external dependencies
**Impact**: Non-blocking context building possible

## Code Organization

### Directory Structure
**Pattern**: Separate core (Node, Context, Cache) from model specializations
**Benefit**: Clean separation of concerns, easy to extend

### Header-Only vs Implementation
**Decision**: Header-only for simplicity (single .hpp files)
**Benefit**: No linking complexity, easy to understand
**Consideration**: May split for larger implementations
