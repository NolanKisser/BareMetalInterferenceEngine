#ifndef SERVER_H
#define SERVER_H

#include "execution_engine.h"
#include <memory>
#include <string>

class InferenceServer {
public:
    InferenceServer(std::shared_ptr<ExecutionEngine> engine, int port);
    ~InferenceServer();

    // Starts the blocking server loop to listen for TCP connections
    void start();

private:
    std::shared_ptr<ExecutionEngine> engine_;
    int port_;

    // Handle a single client request
    void handle_client(const std::string& request_payload);
};

#endif // SERVER_H
