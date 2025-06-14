#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream file("/etc/os-release");
    if (!file) {
        std::cerr << "Failed to open /etc/os-release" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("PRETTY_NAME=") == 0) {
            // Remove PRETTY_NAME= and any surrounding quotes
            size_t start = line.find('=') + 1;
            if (line[start] == '"') start++;
            size_t end = line.find_last_not_of('"');
            std::cout << "Linux Version: " << line.substr(start, end - start + 1) << std::endl;
            break;
        }
    }
    
    return 0;
}

