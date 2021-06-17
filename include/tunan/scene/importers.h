//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_IMPORTERS_H
#define TUNAN_IMPORTERS_H

#include <tunan/common.h>

#include <string>
#include <map>

namespace RENDER_NAMESPACE {
    class SceneData;

    namespace utils {
        class ResourceManager;
    }

    namespace importer {
        using utils::ResourceManager;

        class MitsubaSceneImporter {
        public:
            MitsubaSceneImporter();

            void importScene(std::string sceneDirectory, SceneData &scene, ResourceManager *allocator);
        };
    }
}

#endif //TUNAN_IMPORTERS_H
