//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/tracer/path.h>
#include <tunan/parallel/parallels.h>
#include <tunan/sampler/SamplerFactory.h>

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::SamplerFactory;

        PathTracer::PathTracer(SceneData &parsedScene, MemoryAllocator &allocator) :
                _allocator(allocator), _world(parsedScene, allocator) {
            _filmWidth = parsedScene.width;
            _filmHeight = parsedScene.height;
            _camera = allocator.newObject<Camera>(parsedScene.cameraToWorld, parsedScene.fov, _filmWidth, _filmHeight);
            _sampler = SamplerFactory::newSampler(parsedScene.sampleNum, _allocator);

            // Initialize queues
            int maxQueueSize = _filmWidth * _scanLines;
            _rayQueue = allocator.newObject<RayQueue>(maxQueueSize, allocator);
            _missQueue = allocator.newObject<MissQueue>(maxQueueSize, allocator);
            _mediaEvaQueue = allocator.newObject<MediaEvaQueue>(maxQueueSize, allocator);
            _materialEvaQueue = allocator.newObject<MaterialEvaQueue>(maxQueueSize, allocator);
            _areaLightEvaQueue = allocator.newObject<AreaLightHitQueue>(maxQueueSize, allocator);
            _pixelQueue = allocator.newObject<PixelQueue>(maxQueueSize, allocator);
        }

        void PathTracer::render() {
            for (int sampleIndex = 0; sampleIndex < _nIterations; sampleIndex++) {
                for (int row = 0; row < _filmHeight; row += _scanLines) {
                    // TODO Reset queues
                    _rayQueue->reset();
                    // Generate camera rays
                    generateCameraRays(sampleIndex, row);
                    // TODO Need synchronized ?
                    for (int bounce = 0; bounce < _maxBounce; bounce++) {
                        _world.intersect(_rayQueue, _missQueue, _materialEvaQueue, _mediaEvaQueue, _areaLightEvaQueue);
                        // TODO Handle media queue
                        // TODO Handle area light queue

                        evaluateMissRays(sampleIndex, row);

                        evaluateMateriaAndBSDF(sampleIndex, row);
                    }
                }
            }
        }

        void PathTracer::generateCameraRays(int sampleIndex, int scanLine) {
            auto func = [=](auto sampler) {
                using SamplerType = typename std::remove_reference<decltype(*sampler)>::type;
                return generateCameraRays<SamplerType>(sampleIndex, scanLine);
            };
            _sampler.proxyCall(func);
        }

        template<typename SamplerType>
        void PathTracer::generateCameraRays(int sampleIndex, int scanLine) {
            // TODO refactor cuda declare
            auto func = RENDER_CPU_GPU_LAMBDA(int idx) {
                int pixelY = idx / _filmWidth + scanLine;
                int pixelX = idx % _filmWidth;

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.forPixel(Point2I(pixelX, pixelY));

                // TODO use: generateRayDifferential
                RayDetails &rayDetails = (*_rayQueue)[idx];
                rayDetails.ray = _camera->generateRay(pixelX, pixelY);
                rayDetails.pixelIndex = idx;
            };

            int nItem = _rayQueue->size();
            parallel::parallelFor(func, nItem);
        }
    }
}