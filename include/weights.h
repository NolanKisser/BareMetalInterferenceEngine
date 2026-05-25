#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <string>
#include <vector>
#include <memory>

// Basic tensor structure
struct Tensor {
    std::string name;
    std::vector<int> shape;
    // For a bare-metal approach, we might just store a pointer to float16 or int8
    void* data = nullptr; 
    size_t size_bytes = 0;

    ~Tensor() {
        if (data) {
            free(data);
        }
    }
};

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    bool load_metadata(const std::string& model_path);
    bool load_weights(const std::string& model_path);

    const std::vector<std::shared_ptr<Tensor>>& get_tensors() const;

private:
    std::vector<std::shared_ptr<Tensor>> tensors;
};

#endif // WEIGHTS_H
