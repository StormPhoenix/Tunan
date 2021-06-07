//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/tracer/path.h>
#include <tunan/parallel/parallels.h>
#include <tunan/sampler/samplers.h>
#include <tunan/sampler/SamplerFactory.h>

#include <iostream>

#ifdef __RENDER_GPU_MODE__

#include <tunan/scene/OptixIntersectable.h>

#endif

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::SamplerFactory;

        PathTracer::PathTracer(SceneData &parsedScene, MemoryAllocator &allocator) :
                _allocator(allocator) {
#ifdef __RENDER_GPU_MODE__
            _world = new OptixIntersectable(parsedScene, allocator);
#else
            // TODO cpu scene intersectable
#endif
            _filmWidth = parsedScene.width;
            _filmHeight = parsedScene.height;
            _camera = allocator.newObject<Camera>(parsedScene.cameraToWorld, parsedScene.fov, _filmWidth, _filmHeight);
            _sampler = SamplerFactory::newSampler(parsedScene.sampleNum, _allocator);
            _nIterations = parsedScene.sampleNum;
            _maxBounce = parsedScene.maxDepth;

            // Initialize queues
            _maxQueueSize = _filmWidth * _scanLines;
            _rayQueue = allocator.newObject<RayQueue>(_maxQueueSize, allocator);
            _missQueue = allocator.newObject<MissQueue>(_maxQueueSize, allocator);
            _mediaEvaQueue = allocator.newObject<MediaEvaQueue>(_maxQueueSize, allocator);
            _materialEvaQueue = allocator.newObject<MaterialEvaQueue>(_maxQueueSize, allocator);
            _areaLightEvaQueue = allocator.newObject<AreaLightHitQueue>(_maxQueueSize, allocator);

            _pixelArray = allocator.newObject<PixelStateArray>(allocator);
            _pixelArray->reset(_maxQueueSize);
        }

        void PathTracer::render() {
            for (int sampleIndex = 0; sampleIndex < _nIterations; sampleIndex++) {
                for (int row = 0; row < _filmHeight; row += _scanLines) {
                    // TODO other queues also need resets
                    _rayQueue->reset();
                    generateCameraRays(sampleIndex, row);
                    for (int bounce = 0; bounce < _maxBounce; bounce++) {
                        _missQueue->reset();
                        _materialEvaQueue->reset();
                        generateRaySamples(sampleIndex);
                        _world->intersect(_rayQueue, _missQueue, _materialEvaQueue, _mediaEvaQueue,
                                          _areaLightEvaQueue, _pixelArray);
                        // TODO Handle media queue
                        // TODO Handle area light queue
                        // TODO
//                        evaluateMissRays(sampleIndex, row);
                        // TODO
//                        evaluateMaterialAndBSDF(sampleIndex, row);
                    }
                }
                std::cout << sampleIndex << std::endl;
                _world->writeImage();
            }
        }

        void PathTracer::evaluateMaterialAndBSDF(int sampleIndex, int scanLine) {
            auto func = RENDER_CPU_GPU_LAMBDA(MaterialEvaDetails &m) {
                // TODO
//                m.material.evaluateBSDF()
            };
            parallel::parallelForQueue(func, _materialEvaQueue, _maxQueueSize);
        }

        void PathTracer::generateRaySamples(int sampleIndex) {
            auto func = [=](auto sampler) {
                using SamplerType = typename std::remove_reference<decltype(*sampler)>::type;
                return generateRaySamples<SamplerType>(sampleIndex);
            };
            _sampler.proxyCall(func);
        }

        template<typename SamplerType>
        void PathTracer::generateRaySamples(int sampleIndex) {
            auto func = RENDER_CPU_GPU_LAMBDA(const RayDetails &r) {
                int bounce = r.bounce;
                int pixelIndex = r.pixelIndex;
                // TODO very important
                // 2 for CameraSamples
                int dimension = 2 + 3 * bounce;

                PixelState &pixelState = (*_pixelArray)[pixelIndex];

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.setCurrentSample({pixelState.pixelX, pixelState.pixelY}, sampleIndex, dimension);

                pixelState.raySamples.scatter.u = sampler.sample1D();
                pixelState.raySamples.scatter.uv = sampler.sample2D();
            };
            parallel::parallelForQueue(func, _rayQueue, _maxQueueSize);
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
            auto func = RENDER_CPU_GPU_LAMBDA(int idx) {
                int pixelY = idx / _filmWidth + scanLine;
                int pixelX = idx % _filmWidth;

                if (pixelY < 0 || pixelY >= _filmHeight ||
                    pixelX < 0 || pixelX >= _filmWidth) {
                    return;
                }

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.forPixel(Point2I(pixelX, pixelY));
                sampler.setSampleIndex(sampleIndex);

                CameraSamples cs;
                cs.uvLens = sampler.sample2D();

                RayDetails rayDetails;
                rayDetails.ray = _camera->generateRayDifferential(pixelX, pixelY, cs);
                rayDetails.pixelIndex = idx;
                rayDetails.bounce = 0;
                _rayQueue->enqueue(rayDetails);

                PixelState &pixelState = (*_pixelArray)[idx];
                pixelState.L = Spectrum(0);
                pixelState.pixelX = pixelX;
                pixelState.pixelY = pixelY;
                return;
            };

            parallel::parallelFor(func, _maxQueueSize);
        }

    }
}