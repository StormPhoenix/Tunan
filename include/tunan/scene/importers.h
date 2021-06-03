//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_IMPORTERS_H
#define TUNAN_IMPORTERS_H

#include <tunan/common.h>
#include <tunan/scene/scene_data.h>
#include <tunan/utils/MemoryAllocator.h>

#include <string>

namespace RENDER_NAMESPACE {
    namespace importer {
        using namespace utils;

        class MitsubaSceneImporter {
        public:
            MitsubaSceneImporter();

            void importScene(std::string sceneDirectory, SceneData &scene, MemoryAllocator &allocator);
        };
    }
}

#endif //TUNAN_IMPORTERS_H
