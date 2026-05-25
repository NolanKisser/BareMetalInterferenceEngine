#include <iostream>
#include "tokenizer.h"
#include "weights.h"

int main(int argc, char** argv) {
    std::cout << "BareMetal Inference Engine - Phase 1" << std::endl;

    Tokenizer tokenizer;
    // Create a dummy vocab file for testing
    std::string vocab_path = "dummy_vocab.txt";
    tokenizer.load(vocab_path);

    std::string prompt = "Hello, BareMetal!";
    auto tokens = tokenizer.encode(prompt);
    
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Tokens: [ ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Decoded: " << tokenizer.decode(tokens) << std::endl;

    ModelLoader loader;
    std::string model_path = "dummy_model.bin";
    loader.load_metadata(model_path);
    loader.load_weights(model_path);

    std::cout << "Engine initialization complete." << std::endl;

    return 0;
}
