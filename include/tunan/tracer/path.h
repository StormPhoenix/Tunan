//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_PATH_H
#define TUNAN_PATH_H

#include <tunan/common.h>
#include <tunan/sampler/samplers.h>
#include <tunan/scene/scene_data.h>
#include <tunan/scene/Camera.h>
#include <tunan/base/containers.h>
#include <tunan/base/spectrum.h>
#include <tunan/utils/MemoryAllocator.h>

// TODO move OptixIntersectable.h to *.cpp file
#ifdef __RENDER_GPU_MODE__
#include <tunan/scene/OptixIntersectable.h>
#endif

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::Sampler;

        typedef struct RayDetails {
            Ray ray;
            int pixelIndex;
        } RayDetails;

        typedef struct MaterialEvaDetails {
            // TODO
            int bounce;
            Point3F p;
            Normal3F ng;
            Normal3F ns;
            Vector3F wo;
            int pixelIndex;
        } MaterialEvaDetails;

        typedef struct MediaEvaDetails {
            // TODO
        } MediaEvaDetails;

        typedef struct Pixel {
            // TODO
            Spectrum L;
            int pixelX, pixelY;
        } Pixel;

        typedef struct AreaLightHitDetails {
            // TODO
        } AreaLightHitDetails;

        typedef base::Queue<RayDetails> RayQueue;
        typedef base::Queue<RayDetails> MissQueue;
        typedef base::Queue<MaterialEvaDetails> MaterialEvaQueue;
        typedef base::Queue<MediaEvaDetails> MediaEvaQueue;
        typedef base::Queue<AreaLightHitDetails> AreaLightHitQueue;
        typedef base::Queue<Pixel> PixelQueue;

        class PathTracer {
        public:
            PathTracer(SceneData &parsedScene, MemoryAllocator &allocator);

            void render();

            void generateCameraRays(int sampleIndex, int scanLine);

            template<typename SamplerType>
            void generateCameraRays(int sampleIndex, int scanLine);

            void evaluateMissRays(int sampleIndex, int scanLine);

            void evaluateMateriaAndBSDF(int sampleIndex, int scanLine);

        private:
            Camera *_camera;
            Sampler _sampler;
            int _filmWidth, _filmHeight;
            int _maxBounce;

            // Queues
            RayQueue *_rayQueue;
            MissQueue *_missQueue;
            MaterialEvaQueue *_materialEvaQueue;
            MediaEvaQueue *_mediaEvaQueue;
            AreaLightHitQueue *_areaLightEvaQueue;
            PixelQueue *_pixelQueue;

            // TODO temporay tracing config
            int _scanLines = 100;
            int _nIterations = 100;

            // TODO extract base class {WorldIntersectable}
            OptixIntersectable _world;
            MemoryAllocator &_allocator;
        };
    }
}

#endif //TUNAN_PATH_H
