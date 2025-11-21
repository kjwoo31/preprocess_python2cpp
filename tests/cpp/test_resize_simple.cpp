#include <iostream>
#include "image.h"

int main() {
    // Create a simple 3x3 image
    img::Image src(3, 3, 1);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            src.at(y, x, 0) = (y * 3 + x) * 10;
        }
    }
    
    std::cout << "Source 3x3 image:" << std::endl;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            std::cout << (int)src.at(y, x, 0) << " ";
        }
        std::cout << std::endl;
    }
    
    // Resize to 2x2
    auto dst = img::resize(src, 2, 2);
    
    std::cout << "\nResized 2x2 image:" << std::endl;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            std::cout << (int)dst.at(y, x, 0) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
