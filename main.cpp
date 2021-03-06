#include <iostream>

#include <tunan/scene/importers.h>
#include <tunan/utils/memory/CudaAllocator.h>
#include <tunan/utils/ResourceManager.h>

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

#include <tunan/tracer/path.h>

int main() {
    // TODO delete for testing
    using namespace tunan;
    using namespace tunan::importer;
    using namespace tunan::utils;
    using namespace tunan::tracer;
    using namespace tunan::sampler;

    Allocator *resource = new CudaAllocator();
    ResourceManager allocator(resource);

//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/cornel-box/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/cbox-bunny/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/dragon/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/cbox-bunny-material/";
    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/classroom/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/teapot-full/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/material-testball/";
//    std::string sceneDirectory = fs::current_path().generic_string() + "/resource/scenes/dragon/";
//    std::string sceneDirectory =  "/resource/scenes/cornel-box/";
    MitsubaSceneImporter importer = MitsubaSceneImporter();

    SceneData parsedScene;
    importer.importScene(sceneDirectory, parsedScene, &allocator);

    PathTracer tracer(parsedScene, &allocator);
    tracer.render();
     // TODO think about resource deallocation
//    delete resource;
    return 0;
}
