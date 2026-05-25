#include "tokenizer.h"
#include <iostream>
#include <sstream>

Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
}

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }

    // Dummy logic for now: populate vocab with characters
    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        vocab[line] = id;
        vocab_rev[id] = line;
        id++;
    }
    std::cout << "Loaded " << vocab.size() << " tokens from " << vocab_path << std::endl;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    // Dummy encoding: just return ASCII values for now
    for (char c : text) {
        tokens.push_back(static_cast<int>(c));
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    // Dummy decoding: just cast back from ASCII
    for (int t : tokens) {
        text += static_cast<char>(t);
    }
    return text;
}
