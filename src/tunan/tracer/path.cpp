//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/tracer/path.h>
#include <tunan/parallel/parallels.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/materials.h>
#include <tunan/sampler/samplers.h>
#include <tunan/sampler/SamplerFactory.h>
#include <tunan/utils/type_utils.h>

#include <iostream>

#ifdef __BUILD_GPU_RENDER_ENABLE__

#include <tunan/scene/OptixIntersectable.h>

#endif

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::SamplerFactory;
        using namespace bsdf;

        PathTracer::PathTracer(SceneData &parsedScene, MemoryAllocator &allocator) :
                _allocator(allocator) {
#ifdef __BUILD_GPU_RENDER_ENABLE__
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
            _rayQueues[0] = allocator.newObject<RayQueue>(_maxQueueSize, allocator);
            _rayQueues[1] = allocator.newObject<RayQueue>(_maxQueueSize, allocator);
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
                    currentRayQueue(0)->reset();
                    generateCameraRays(sampleIndex, row);
                    for (int bounce = 0; bounce < _maxBounce; bounce++) {
                        _missQueue->reset();
                        _materialEvaQueue->reset();
                        nextRayQueue(bounce)->reset();

                        generateRaySamples(sampleIndex, bounce);
                        _world->intersect(currentRayQueue(bounce), _missQueue, _materialEvaQueue,
                                          _mediaEvaQueue, _areaLightEvaQueue, _pixelArray);
                        // TODO Handle media queue
                        // TODO Handle area light queue
                        // TODO
//                        evaluateMissRays(sampleIndex, row);
                        // TODO
                        evaluateMaterialBSDF(bounce);
                    }
                }
                std::cout << sampleIndex << std::endl;
                _world->writeImage();
            }
        }

        typedef struct MaterialEvaWrapper {
            template<typename T>
            void operator()() {
                tracer->evaluateMaterialBSDF<T>(bounce);
            }

            PathTracer *tracer;
            int bounce;
        } MaterialEvaWrapper;

        void PathTracer::evaluateMaterialBSDF(int bounce) {
            forEachType(MaterialEvaWrapper({this, bounce}), Material::Types());
        }

        template<typename MaterialType>
        void PathTracer::evaluateMaterialBSDF(int bounce) {
            RayQueue *nextQueue = nextRayQueue(bounce);
            auto func = RENDER_CPU_GPU_LAMBDA(MaterialEvaDetails &m) {
                if (!m.material.isType<MaterialType>()) {
                    return;
                }

                MaterialType *material = m.material.cast<MaterialType>();
                using MaterialBxDF = typename MaterialType::MaterialBxDF;
                MaterialBxDF bxdf;
                BSDF bsdf = material->evaluateBSDF(m.si, &bxdf);

                if (bsdf.allIncludeOf(BxDFType(BSDF_ALL & (!BSDF_SPECULAR)))) {
                    // TODO sample from light
                }

//                printf("ss");
                // sample direction
                PixelState &state = (*_pixelArray)[m.pixelIndex];
                Vector3F wi;
                Float pdf;
                BxDFType sampleType;
                Spectrum f = bsdf.sampleF(m.si.wo, &wi, &pdf, state.raySamples.scatter.uv, &sampleType);
                if (f.isBlack() || pdf == 0.) {
                    return;
                }

                // evaluate beta
                Float cosTheta = ABS_DOT(m.si.ns, NORMALIZE(wi));
                state.beta *= (f * cosTheta / pdf);

                RayDetails r;
                r.pixelIndex = m.pixelIndex;
                r.bounce = m.bounce + 1;
                r.specularBounce = (sampleType & BSDF_SPECULAR) > 0;
                r.ray = m.si.generateRay(wi);
                nextQueue->enqueue(r);
            };
            parallel::parallelForQueue(func, _materialEvaQueue, _maxQueueSize);
        }

        void PathTracer::generateRaySamples(int sampleIndex, int bounce) {
            auto func = [=](auto sampler) {
                using SamplerType = typename std::remove_reference<decltype(*sampler)>::type;
                return generateRaySamples<SamplerType>(sampleIndex, bounce);
            };
            _sampler.proxyCall(func);
        }

        template<typename SamplerType>
        void PathTracer::generateRaySamples(int sampleIndex, int bounce) {
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
            parallel::parallelForQueue(func, currentRayQueue(bounce), _maxQueueSize);
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
            RayQueue *rayQueue = currentRayQueue(0);
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
                rayDetails.specularBounce = false;
                rayQueue->enqueue(rayDetails);

                PixelState &pixelState = (*_pixelArray)[idx];
                pixelState.L = Spectrum(0);
                pixelState.beta = Spectrum(1.0);
                pixelState.pixelX = pixelX;
                pixelState.pixelY = pixelY;
                return;
            };
            parallel::parallelFor(func, _maxQueueSize);
        }

        RayQueue *PathTracer::currentRayQueue(int depth) {
            return _rayQueues[depth % 2];
        }

        RayQueue *PathTracer::nextRayQueue(int depth) {
            return _rayQueues[(depth + 1) % 2];
        }
    }
}