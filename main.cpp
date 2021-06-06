#include <iostream>

// TODO for testing
#include <tunan/common.h>
#include <tunan/scene/importers.h>
#include <tunan/utils/memory/CUDAResource.h>
#include <tunan/utils/MemoryAllocator.h>

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<filesystem>)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <ext/ghc/filesystem.hpp>

namespace fs = ghc::filesystem;
#endif

// TODO delete for testing
#include <tunan/tracer/path.h>
#include <tunan/sampler/SamplerFactory.h>

int main() {
    // TODO delete for testing
    using namespace tunan;
    using namespace tunan::importer;
    using namespace tunan::utils;
    using namespace tunan::tracer;
    using namespace tunan::sampler;

    MemoryResource *resource = new CUDAResource();
    MemoryAllocator allocator(resource);

    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/cornel-box/";
//    std::string sceneDirectory =  "/resource/scenes/cornel-box/";
    MitsubaSceneImporter importer = MitsubaSceneImporter();

    SceneData parsedScene;
    importer.importScene(sceneDirectory, parsedScene, allocator);

    PathTracer tracer(parsedScene, allocator);
    tracer.render();
     // TODO think about resource deallocation
//    delete resource;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
