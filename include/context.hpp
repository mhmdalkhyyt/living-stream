#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <unordered_map>
#include <string>

/**
 * Immutable built context containing model data.
 * Provides getters/setters for all fields.
 */
template<typename T>
class Context {
public:
    Context() = default;
    ~Context() = default;

    // Non-copyable, movable
    Context(const Context&) = default;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;

    // Weight accessors
    std::vector<T> getWeights() const;
    void setWeights(const std::vector<T>& weights);

    // Configuration accessors
    std::unordered_map<std::string, double> getConfig() const;
    void setConfig(const std::unordered_map<std::string, double>& config);

    // Metadata accessors
    std::string getMetadata(const std::string& key) const;
    void setMetadata(const std::string& key, const std::string& value);

private:
    std::vector<T> weights_;
    std::unordered_map<std::string, double> config_;
    std::unordered_map<std::string, std::string> metadata_;
};

#endif // CONTEXT_HPP
