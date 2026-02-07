#include "llm_node.hpp"
#include "context.hpp"

template<typename T>
LLMNode<T>::LLMNode(int slot_index, const std::string& model_path)
    : Node<T>(slot_index)
    , model_path_(model_path)
    , parameter_count_(0) {
}

template<typename T>
std::unique_ptr<Context<T>> LLMNode<T>::buildContext() {
    // TODO: Implement - deep copy model data from slot
    return nullptr;
}

template<typename T>
std::future<std::unique_ptr<Context<T>>> LLMNode<T>::asyncBuildContext() {
    // TODO: Implement with std::async
    return std::future<std::unique_ptr<Context<T>>>();
}

template<typename T>
void LLMNode<T>::loadModel(const std::string& path) {
    // TODO: Implement model loading
}

template<typename T>
int LLMNode<T>::getParameterCount() const {
    return parameter_count_;
}

// Explicit template instantiations
template class LLMNode<float>;
template class LLMNode<double>;
