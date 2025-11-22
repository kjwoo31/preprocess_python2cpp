#include <iostream>
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct StbImageDeleter {
    void operator()(unsigned char* ptr) const {
        stbi_image_free(ptr);
    }
};

int main() {
    const std::string path = "config/data/test_image.jpg";

    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<unsigned char, StbImageDeleter> data(
        stbi_load(path.c_str(), &width, &height, &channels, 0)
    );

    if (!data) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Loaded: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "First pixel (RGB): ["
              << static_cast<int>(data.get()[2]) << ", "
              << static_cast<int>(data.get()[1]) << ", "
              << static_cast<int>(data.get()[0]) << "]" << std::endl;
    std::cout << "Expected (BGR): [225, 189, 159]" << std::endl;

    return 0;
}
