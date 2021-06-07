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
        class MemoryAllocator;
    }

    namespace importer {
        using utils::MemoryAllocator;

        class MitsubaSceneImporter {
        public:
            MitsubaSceneImporter();

            void importScene(std::string sceneDirectory, SceneData &scene, MemoryAllocator &allocator);
        };
    }
}

#endif //TUNAN_IMPORTERS_H
