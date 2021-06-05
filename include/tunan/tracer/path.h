//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_PATH_H
#define TUNAN_PATH_H

#include <tunan/common.h>
#include <tunan/sampler/samplers.h>
#include <tunan/scene/Camera.h>
#include <tunan/scene/scene_data.h>
#include <tunan/scene/OptixIntersectable.h>
#include <tunan/base/containers.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::Sampler;

        class PathTracer {
        public:
            PathTracer(SceneData &parsedScene, MemoryAllocator &allocator);

            void render();

            void generateCameraRays(int sampleIndex, int scanLine);

            template<typename SamplerType>
            void generateCameraRays(int sampleIndex, int scanLine);

        private:
            Camera *_camera;
            Sampler _sampler;
            int _filmWidth, _filmHeight;
            base::Vector<Ray> _rayQueues;

            // TODO temporay tracing config
            int _scanLines = 100;
            int _nIterations = 100;

            OptixIntersectable _world;
            MemoryAllocator &_allocator;
        };
    }
}

#endif //TUNAN_PATH_H
