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
                _allocator(allocator), _world(parsedScene, allocator), _rayQueues(allocator) {
            _filmWidth = parsedScene.width;
            _filmHeight = parsedScene.height;
            _camera = allocator.newObject<Camera>(parsedScene.cameraToWorld, parsedScene.fov, _filmWidth, _filmHeight);
            _sampler = SamplerFactory::newSampler(parsedScene.sampleNum, _allocator);
        }

        void PathTracer::render() {
            for (int sampleIndex = 0; sampleIndex < _nIterations; sampleIndex++) {
                for (int row = 0; row < _filmHeight; row += _scanLines) {
                    int scanLines = (row + _scanLines) > _filmHeight ? _filmHeight - row : _scanLines;
                    int traceRayCount = _filmWidth * scanLines;
                    _rayQueues.reset(traceRayCount);

                    // Generate camera rays
                    generateCameraRays(sampleIndex, row);
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
            auto func = [=] __host__ __device__ (int idx) mutable {
                int pixelY = idx / _filmWidth + scanLine;
                int pixelX = idx % _filmWidth;

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.forPixel(Point2I(pixelX, pixelY));

                // TODO use: generateRayDifferential
                _rayQueues[idx] = _camera->generateRay(pixelX, pixelY);
            };

            int nItem = _rayQueues.size();
            parallel::parallelFor(func, nItem);
        }
    }
}