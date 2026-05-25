#include "weights.h"
#include <iostream>
#include <fstream>
#include <cstring>

ModelLoader::ModelLoader() {
}

ModelLoader::~ModelLoader() {
}

bool ModelLoader::load_metadata(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }

    // Dummy metadata loading: imagine reading a header
    std::cout << "Loading metadata from " << model_path << std::endl;
    
    // Create a dummy tensor metadata
    auto tensor = std::make_shared<Tensor>();
    tensor->name = "tok_embeddings.weight";
    tensor->shape = {32000, 4096}; 
    tensor->size_bytes = 32000 * 4096 * sizeof(float); // Dummy 32-bit float size
    tensors.push_back(tensor);

    std::cout << "Found tensor: " << tensor->name << " with shape [" << tensor->shape[0] << ", " << tensor->shape[1] << "]" << std::endl;

    return true;
}

bool ModelLoader::load_weights(const std::string& model_path) {
    // In a real implementation, we would `mmap` the file or read chunks into memory
    std::cout << "Loading weights from " << model_path << "..." << std::endl;
    for (auto& tensor : tensors) {
        tensor->data = malloc(tensor->size_bytes);
        // Dummy data fill
        memset(tensor->data, 0, tensor->size_bytes);
        std::cout << "Allocated " << tensor->size_bytes << " bytes for " << tensor->name << std::endl;
    }
    return true;
}

const std::vector<std::shared_ptr<Tensor>>& ModelLoader::get_tensors() const {
    return tensors;
}
