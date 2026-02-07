
#ifndef NODE_HPP
#define NODE_HPP

#include <future>
#include <string>

template<typename T>
class Context;

/**
 * Abstract base class for all model nodes.
 * Provides interface for building contexts from model slots.
 */
template<typename T>
class Node {
public:
    explicit Node(int slot_index);
    virtual ~Node() = default;

    // Non-copyable, movable
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = default;
    Node& operator=(Node&&) = default;

    // Context building
    virtual std::unique_ptr<Context<T>> buildContext() = 0;
    virtual std::future<std::unique_ptr<Context<T>>> asyncBuildContext() = 0;

    // Accessors
    int getSlotIndex() const;

protected:
    int slot_index_;
};

#endif // NODE_HPP
