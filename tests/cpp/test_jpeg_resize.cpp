#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    std::string path = "config/data/test_image.jpg";
    
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "First pixel (RGB): [" << (int)data[2] << ", " << (int)data[1] << ", " << (int)data[0] << "]" << std::endl;
    std::cout << "Expected (BGR): [225, 189, 159]" << std::endl;
    
    stbi_image_free(data);
    return 0;
}
