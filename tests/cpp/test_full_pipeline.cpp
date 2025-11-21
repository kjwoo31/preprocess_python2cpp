#include <iostream>
#include "image.h"

int main() {
    std::string path = "/mnt/c/Users/kimjw/Downloads/preprocess_python2cpp/config/data/test_image.jpg";
    
    std::cout << "=== Step 1: imread ===" << std::endl;
    auto img1 = img::imread(path);
    std::cout << "Shape: (" << img1.height << ", " << img1.width << ", " << img1.channels << ")" << std::endl;
    std::cout << "First pixel: [" << (int)img1.at(0, 0, 0) << ", " << (int)img1.at(0, 0, 1) << ", " << (int)img1.at(0, 0, 2) << "]" << std::endl;
    std::cout << "Pixel (100,100): [" << (int)img1.at(100, 100, 0) << ", " << (int)img1.at(100, 100, 1) << ", " << (int)img1.at(100, 100, 2) << "]" << std::endl;
    
    std::cout << "\n=== Step 2: resize (new variable) ===" << std::endl;
    auto img2 = img::resize(img1, 224, 224);
    std::cout << "Shape: (" << img2.height << ", " << img2.width << ", " << img2.channels << ")" << std::endl;
    std::cout << "First pixel: [" << (int)img2.at(0, 0, 0) << ", " << (int)img2.at(0, 0, 1) << ", " << (int)img2.at(0, 0, 2) << "]" << std::endl;
    std::cout << "Pixel (100,100): [" << (int)img2.at(100, 100, 0) << ", " << (int)img2.at(100, 100, 1) << ", " << (int)img2.at(100, 100, 2) << "]" << std::endl;
    
    std::cout << "\n=== Step 3: resize (same variable via assignment) ===" << std::endl;
    auto img3 = img::imread(path);
    std::cout << "Before: [" << (int)img3.at(0, 0, 0) << ", " << (int)img3.at(0, 0, 1) << ", " << (int)img3.at(0, 0, 2) << "]" << std::endl;
    img3 = img::resize(img3, 224, 224);
    std::cout << "After: [" << (int)img3.at(0, 0, 0) << ", " << (int)img3.at(0, 0, 1) << ", " << (int)img3.at(0, 0, 2) << "]" << std::endl;
    std::cout << "Shape: (" << img3.height << ", " << img3.width << ", " << img3.channels << ")" << std::endl;
    
    return 0;
}
