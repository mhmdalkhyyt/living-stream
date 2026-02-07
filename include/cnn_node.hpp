#ifndef CNN_NODE_HPP
#define CNN_NODE_HPP

#include "node.hpp"
#include <string>

template<typename T>
class Context;

/**
 * CNN-specific node specialization.
 */
template<typename T>
class CNNNode : public Node<T> {
public:
    explicit CNNNode(int slot_index, const std::string& model_path = "");
    ~CNNNode() override = default;

    // Context building
    std::unique_ptr<Context<T>> buildContext() override;
    std::future<std::unique_ptr<Context<T>>> asyncBuildContext() override;

    // CNN-specific methods
    void loadModel(const std::string& path);
    int getLayerCount() const;
    std::vector<int> getLayerSizes() const;

private:
    std::string model_path_;
    int layer_count_;
    std::vector<int> layer_sizes_;
};

#endif // CNN_NODE_HPP
