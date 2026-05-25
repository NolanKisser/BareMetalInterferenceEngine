#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> vocab_rev;
    // BPE merges could be added here
};

#endif // TOKENIZER_H
