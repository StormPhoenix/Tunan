#include <iostream>

// TODO for testing
#include <tunan/scene/OptiXScene.h>

int main() {
    // TODO for testing
    using namespace tunan;
    OptiXScene scene;
    scene.buildOptiXData();
    scene.intersect();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
