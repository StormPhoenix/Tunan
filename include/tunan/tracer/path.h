//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_PATH_H
#define TUNAN_PATH_H

#include <tunan/common.h>
#include <tunan/sampler/samplers.h>
#include <tunan/scene/Camera.h>
#include <tunan/scene/scene_data.h>
#include <tunan/scene/SceneIntersectable.h>
#include <tunan/utils/MemoryAllocator.h>
#include <tunan/tracer/tracer.h>

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

            void generateRaySamples(int sampleIndex);

            template<typename SamplerType>
            void generateRaySamples(int sampleIndex);

            void evaluateMissRays(int sampleIndex, int scanLine);

            void evaluateMaterialAndBSDF(int sampleIndex, int scanLine);

        private:
            Camera *_camera;
            Sampler _sampler;
            int _filmWidth, _filmHeight;
            int _maxQueueSize = 0;
            int _maxBounce;

            // Queues
            RayQueue *_rayQueue;
            MissQueue *_missQueue;
            MaterialEvaQueue *_materialEvaQueue;
            MediaEvaQueue *_mediaEvaQueue;
            AreaLightHitQueue *_areaLightEvaQueue;
            PixelStateArray *_pixelArray;

            // TODO temporay tracing config
            int _scanLines = 100;
            int _nIterations = 100;

            // TODO extract base class {WorldIntersectable}
            SceneIntersectable *_world;
            MemoryAllocator &_allocator;
        };
    }
}

#endif //TUNAN_PATH_H
