#ifndef LLM_NODE_HPP
#define LLM_NODE_HPP

#include "node.hpp"
#include <string>

template<typename T>
class Context;

/**
 * LLM-specific node specialization.
 */
template<typename T>
class LLMNode : public Node<T> {
public:
    explicit LLMNode(int slot_index, const std::string& model_path = "");
    ~LLMNode() override = default;

    // Context building
    std::unique_ptr<Context<T>> buildContext() override;
    std::future<std::unique_ptr<Context<T>>> asyncBuildContext() override;

    // LLM-specific methods
    void loadModel(const std::string& path);
    int getParameterCount() const;

private:
    std::string model_path_;
    int parameter_count_;
};

#endif // LLM_NODE_HPP
