#include "cnn_node.hpp"
#include "context.hpp"

template<typename T>
CNNNode<T>::CNNNode(int slot_index, const std::string& model_path)
    : Node<T>(slot_index)
    , model_path_(model_path)
    , layer_count_(0) {
}

template<typename T>
std::unique_ptr<Context<T>> CNNNode<T>::buildContext() {
    // TODO: Implement - deep copy model data from slot
    return nullptr;
}

template<typename T>
std::future<std::unique_ptr<Context<T>>> CNNNode<T>::asyncBuildContext() {
    // TODO: Implement with std::async
    return std::future<std::unique_ptr<Context<T>>>();
}

template<typename T>
void CNNNode<T>::loadModel(const std::string& path) {
    // TODO: Implement model loading
}

template<typename T>
int CNNNode<T>::getLayerCount() const {
    return layer_count_;
}

template<typename T>
std::vector<int> CNNNode<T>::getLayerSizes() const {
    return layer_sizes_;
}

// Explicit template instantiations
template class CNNNode<float>;
template class CNNNode<double>;
