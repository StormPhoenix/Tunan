//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_PATH_H
#define TUNAN_PATH_H

#include <tunan/common.h>
#include <tunan/sampler/samplers.h>
#include <tunan/scene/film.h>
#include <tunan/scene/cameras.h>
#include <tunan/scene/scenedata.h>
#include <tunan/scene/SceneIntersectable.h>
#include <tunan/utils/ResourceManager.h>
#include <tunan/tracer/tracer.h>

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::Sampler;

        typedef struct PTParameters {
            std::string filename;
            int scanLines = 400;
            int nIterations = 100;
            int maxBounce = 12;
            int filmWidth, filmHeight;
            int writeFrequency = 50;
        } PTParameters;

        class PathTracer {
        public:
            PathTracer(SceneData &parsedScene, ResourceManager &allocator);

            void render();

            void generateCameraRays(int sampleIndex, int scanLine);

            template<typename SamplerType>
            void generateCameraRays(int sampleIndex, int scanLine);

            void generateRaySamples(int sampleIndex, int bounce);

            template<typename SamplerType>
            void generateRaySamples(int sampleIndex, int bounce);

            void evaluateMaterialBSDF(int bounce);

            template<typename MaterialType>
            void evaluateMaterialBSDF(int bounce);

            void evaluateAreaLightQueue();

            void evaluateMissRays(int sampleIndex, int scanLine);

            void updateFilm(int nCameraRays);

        protected:
            RayQueue *currentRayQueue(int depth);

            RayQueue *nextRayQueue(int depth);

        private:
            Film *_film = nullptr;
            Camera *_camera = nullptr;
            Sampler _sampler;
            int _maxQueueSize = 0;

            // Pixels
            PixelStateArray *_pixelArray;
            // Lights
            base::Vector<Light> *_lights;

            // Queues
            RayQueue *_rayQueues[2];
            MissQueue *_missQueue;
            MediaEvaQueue *_mediaEvaQueue;
            MaterialEvaQueue *_materialEvaQueue;
            AreaLightHitQueue *_areaLightEvaQueue;
            ShadowRayQueue *_shadowRayQueue;

            PTParameters params;

            SceneIntersectable *_world;
            ResourceManager &_allocator;
        };
    }
}

#endif //TUNAN_PATH_H
