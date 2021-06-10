//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/tracer/path.h>
#include <tunan/parallel/parallels.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/materials.h>
#include <tunan/sampler/samplers.h>
#include <tunan/sampler/SamplerFactory.h>
#include <tunan/scene/lights.h>
#include <tunan/utils/type_utils.h>

#include <iostream>
#include <sstream>

#ifdef __BUILD_GPU_RENDER_ENABLE__

#include <tunan/scene/OptixIntersectable.h>

#endif

namespace RENDER_NAMESPACE {
    namespace tracer {
        using sampler::SamplerFactory;
        using namespace bsdf;

        void printDetails(PTParameters &params) {
            // Print config info
            std::cout << std::endl << "Using render type: PT" << std::endl;
            std::cout << "Max depth: " << params.maxBounce << std::endl;
            std::cout << "Sample number: " << params.nIterations << std::endl;
            std::cout << "Size: (" << params.filmWidth << ", " << params.filmHeight << ")" << std::endl;
        }

        PathTracer::PathTracer(SceneData &parsedScene, MemoryAllocator &allocator) :
                _allocator(allocator) {
#ifdef __BUILD_GPU_RENDER_ENABLE__
            _world = new OptixIntersectable(parsedScene, allocator);
#else
            // TODO cpu scene intersectable
#endif
            params.filmWidth = parsedScene.width;
            params.filmHeight = parsedScene.height;
            params.maxBounce = parsedScene.maxDepth;
            params.nIterations = parsedScene.sampleNum;
            params.filename = parsedScene.filename;

            _camera = allocator.newObject<Camera>(parsedScene.cameraToWorld, parsedScene.fov,
                                                  params.filmWidth, params.filmHeight);
            _sampler = SamplerFactory::newSampler(params.nIterations, _allocator);


            // Initialize queues
            _maxQueueSize = params.filmWidth * params.scanLines;
            _rayQueues[0] = allocator.newObject<RayQueue>(_maxQueueSize, allocator);
            _rayQueues[1] = allocator.newObject<RayQueue>(_maxQueueSize, allocator);
            _missQueue = allocator.newObject<MissQueue>(_maxQueueSize, allocator);
            _mediaEvaQueue = allocator.newObject<MediaEvaQueue>(_maxQueueSize, allocator);
            _materialEvaQueue = allocator.newObject<MaterialEvaQueue>(_maxQueueSize, allocator);
            _areaLightEvaQueue = allocator.newObject<AreaLightHitQueue>(_maxQueueSize, allocator);
            _shadowRayQueue = allocator.newObject<ShadowRayQueue>(_maxQueueSize, allocator);

            _pixelArray = allocator.newObject<PixelStateArray>(allocator);
            _pixelArray->reset(_maxQueueSize);

            _film = allocator.newObject<Film>(params.filmWidth, params.filmHeight, allocator);
            _lights = parsedScene.lights;
        }

        void PathTracer::render() {
            printDetails(params);
            for (int sampleIndex = 0; sampleIndex < params.nIterations; sampleIndex++) {
                for (int row = 0; row < params.filmHeight; row += params.scanLines) {
                    // TODO other queues also need resets
                    currentRayQueue(0)->reset();
                    generateCameraRays(sampleIndex, row);
                    int nCameraRays = currentRayQueue(0)->size();
                    for (int bounce = 0; bounce < params.maxBounce; bounce++) {
                        if (currentRayQueue(bounce)->size() == 0) {
                            break;
                        }
                        _missQueue->reset();
                        _materialEvaQueue->reset();
                        _areaLightEvaQueue->reset();
                        _shadowRayQueue->reset();
                        nextRayQueue(bounce)->reset();

                        generateRaySamples(sampleIndex, bounce);
                        _world->findClosestHit(currentRayQueue(bounce), _missQueue, _materialEvaQueue,
                                               _mediaEvaQueue, _areaLightEvaQueue, _pixelArray);
                        // TODO Handle media queue
                        // TODO Handle missing queue
//                        evaluateMissRays(sampleIndex, row);
                        evaluateAreaLightQueue();
                        evaluateMaterialBSDF(bounce);
                        _world->traceShadowRay(_shadowRayQueue, _pixelArray);
                    }
                    updateFilm(nCameraRays);
                }

                // Write image frequently
                if ((params.writeFrequency > 0 && (sampleIndex + 1) % params.writeFrequency == 0)) {
                    Float sampleWeight = 1.0 / (sampleIndex + 1);
                    std::string prefix;
                    std::stringstream ss;
                    ss << "SSP" << sampleIndex + 1 << "_";
                    ss >> prefix;
                    _film->writeImage((prefix + params.filename).c_str(), sampleWeight);
                    std::cout << "\r" << float(sampleIndex + 1) * 100 / (params.nIterations) << " %" << std::flush;
                }
            }
            _film->writeImage(params.filename.c_str(), 1.0 / params.nIterations);
        }

        void PathTracer::updateFilm(int nCameraRays) {
            auto func = RENDER_CPU_GPU_LAMBDA(int idx) {
                PixelState &state = (*_pixelArray)[idx];
                int pixelX = state.pixelX;
                int pixelY = state.pixelY;
                _film->addSpectrum(state.L, pixelY, pixelX);
            };
            parallel::parallelFor(func, nCameraRays);
        }

        typedef struct MaterialEvaWrapper {
            template<typename T>
            void operator()() {
                tracer->evaluateMaterialBSDF<T>(bounce);
            }

            PathTracer *tracer;
            int bounce;
        } MaterialEvaWrapper;

        void PathTracer::evaluateAreaLightQueue() {
            auto func = RENDER_CPU_GPU_LAMBDA(AreaLightHitDetails & m) {
                if (m.bounce == 0 || m.specularBounce) {
                    Spectrum L = m.areaLight->L(m.si, m.si.wo);
                    PixelState &state = (*_pixelArray)[m.pixelIndex];
                    state.L += state.beta * L;
                }
            };
            parallel::parallelForQueue(func, _areaLightEvaQueue, _maxQueueSize);
        }

        void PathTracer::evaluateMaterialBSDF(int bounce) {
            forEachType(MaterialEvaWrapper({this, bounce}), Material::Types());
        }

        template<typename MaterialType>
        void PathTracer::evaluateMaterialBSDF(int bounce) {
            RayQueue *nextQueue = nextRayQueue(bounce);
            auto func = RENDER_CPU_GPU_LAMBDA(MaterialEvaDetails & m) {
                if (!m.material.isType<MaterialType>()) {
                    return;
                }

                PixelState &state = (*_pixelArray)[m.pixelIndex];

                MaterialType *material = m.material.cast<MaterialType>();
                using MaterialBxDF = typename MaterialType::MaterialBxDF;
                MaterialBxDF bxdf;

                BSDF bsdf = material->evaluateBSDF(m.si, &bxdf);
                if (bsdf.allIncludeOf(BxDFType(BSDF_ALL & (~BSDF_SPECULAR)))) {
                    // TODO sample from light
                    int nLights = _lights->size();
                    if (nLights != 0) {
                        Float lightPdf = Float(1.0) / nLights;
                        int lightIdx = int(state.raySamples.sampleLight.light * nLights) < (nLights - 1) ?
                                       int(state.raySamples.sampleLight.light * nLights) : (nLights - 1);
                        Light &light = (*_lights)[lightIdx];
                        // Do sampling
                        Vector3F lightDirection;
                        Float samplePdf = 0;
                        Interaction target;
                        Spectrum Li = light.sampleLi(m.si, &lightDirection, &samplePdf,
                                                     state.raySamples.sampleLight.uv, &target);

                        if (!Li.isBlack() && samplePdf > 0.) {
                            // TODO handle medium
                            Float cosTheta = ABS_DOT(m.si.ns, lightDirection);
                            ShadowRayDetails details;
                            details.beta = state.beta * bsdf.f(m.si.wo, lightDirection) * cosTheta;
                            details.scatterPdf = bsdf.samplePdf(m.si.wo, lightDirection);
                            details.sampleLightPdf = samplePdf;
                            details.deltaType = light.isDeltaType();
                            details.pixelIndex = m.pixelIndex;
                            details.L = Li;
                            details.ray = m.si.generateRayTo(target);
                            details.tMax = details.ray.getStep();
                            details.lightPdf = lightPdf;
                            _shadowRayQueue->enqueue(details);
                        }
                    }
                }

                Vector3F wi;
                Float pdf;
                BxDFType sampleType;
                // sample direction
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
                return generateRaySamples < SamplerType > (sampleIndex, bounce);
            };
            _sampler.proxyCall(func);
        }

        template<typename SamplerType>
        void PathTracer::generateRaySamples(int sampleIndex, int bounce) {
            auto func = RENDER_CPU_GPU_LAMBDA(
            const RayDetails &r) {
                int bounce = r.bounce;
                int pixelIndex = r.pixelIndex;
                // TODO very important
                // 4 for CameraSamples
                int dimension = 4 + (3 + 3) * bounce;

                PixelState &pixelState = (*_pixelArray)[pixelIndex];

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.setCurrentSample({pixelState.pixelX, pixelState.pixelY}, sampleIndex, dimension);

                pixelState.raySamples.scatter.u = sampler.sample1D();
                pixelState.raySamples.scatter.uv = sampler.sample2D();

                pixelState.raySamples.sampleLight.light = sampler.sample1D();
                pixelState.raySamples.sampleLight.uv = sampler.sample2D();
            };
            parallel::parallelForQueue(func, currentRayQueue(bounce), _maxQueueSize);
        }

        void PathTracer::generateCameraRays(int sampleIndex, int scanLine) {
            auto func = [=](auto sampler) {
                using SamplerType = typename std::remove_reference<decltype(*sampler)>::type;
                return generateCameraRays < SamplerType > (sampleIndex, scanLine);
            };
            _sampler.proxyCall(func);
        }

        template<typename SamplerType>
        void PathTracer::generateCameraRays(int sampleIndex, int scanLine) {
            RayQueue *rayQueue = currentRayQueue(0);
            auto func = RENDER_CPU_GPU_LAMBDA(int idx) {
                int pixelY = idx / params.filmWidth + scanLine;
                int pixelX = idx % params.filmWidth;

                if (pixelY < 0 || pixelY >= params.filmHeight ||
                    pixelX < 0 || pixelX >= params.filmWidth) {
                    return;
                }

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.forPixel(Point2I(pixelX, pixelY));
                sampler.setSampleIndex(sampleIndex);

                CameraSamples cs;
                cs.uvLens = sampler.sample2D();
                cs.pixelJitter = sampler.sample2D();

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