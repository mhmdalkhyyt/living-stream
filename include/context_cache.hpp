#ifndef CONTEXT_CACHE_HPP
#define CONTEXT_CACHE_HPP

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <list>

template<typename T>
class Context;

/**
 * LRU-style cache for built contexts.
 * Thread-safe using shared_mutex.
 */
template<typename T>
class ContextCache {
public:
    ContextCache() = default;
    ~ContextCache() = default;

    // Non-copyable
    ContextCache(const ContextCache&) = delete;
    ContextCache& operator=(const ContextCache&) = delete;

    // Cache operations
    void cacheContext(int id, std::unique_ptr<Context<T>> context);
    std::unique_ptr<Context<T>> getCachedContext(int id);
    void removeContext(int id);
    void clear();

    // Cache info
    size_t size() const;
    bool contains(int id) const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<int, std::list<std::pair<int, std::unique_ptr<Context<T>>>>> cache_;
};

#endif // CONTEXT_CACHE_HPP
