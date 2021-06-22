//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/base/hash.h>
#include <tunan/tracer/path.h>
#include <tunan/parallel/parallels.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/materials.h>
#include <tunan/medium/mediums.h>
#include <tunan/sampler/rng.h>
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
        using namespace base;
        using namespace bsdf;
        using namespace sampler;

        void printDetails(PTParameters &params) {
            // Print config info
            std::cout << std::endl << "Using render type: PT" << std::endl;
            std::cout << "Max depth: " << params.maxBounce << std::endl;
            std::cout << "Sample number: " << params.nIterations << std::endl;
            std::cout << "Size: (" << params.filmWidth << ", " << params.filmHeight << ")" << std::endl;
        }

        RENDER_CPU_GPU
        inline Spectrum
        sampleDirectLight(base::Vector <Light> *lights, const Interaction &eye, LightSample &lightSample,
                          Vector3F &shadowRayDirection, Float &lightPdf, Float &samplePdf,
                          Interaction &target, bool &isDeltaLight) {
            int nLights = lights->size();
            if (nLights != 0) {
                lightPdf = Float(1.0) / nLights;
                int lightIdx = int(lightSample.light * nLights) < (nLights - 1) ?
                               int(lightSample.light * nLights) : (nLights - 1);
                Light &light = (*lights)[lightIdx];
                isDeltaLight = light.isDeltaType();
                return light.sampleLi(eye, &shadowRayDirection, &samplePdf, lightSample.uv, &target);
            } else {
                return Spectrum(0.f);
            }
        }

        PathTracer::PathTracer(SceneData &parsedScene, ResourceManager *resourceManager) :
                _resourceManager(resourceManager) {
#ifdef __BUILD_GPU_RENDER_ENABLE__
            _world = new OptixIntersectable(parsedScene, resourceManager);
#else
            // TODO cpu scene intersectable
#endif
            params.filmWidth = parsedScene.width;
            params.filmHeight = parsedScene.height;
            params.maxBounce = parsedScene.maxDepth;
            params.nIterations = parsedScene.sampleNum;
            params.filename = parsedScene.filename;

            _camera = resourceManager->newObject<Camera>(parsedScene.cameraToWorld, parsedScene.fov,
                                                         params.filmWidth, params.filmHeight);
            _sampler = SamplerFactory::newSampler(params.nIterations, _resourceManager);


            // Initialize queues
            _maxQueueSize = params.filmWidth * params.scanLines;
            _rayQueues[0] = resourceManager->newObject<RayQueue>(_maxQueueSize, resourceManager);
            _rayQueues[1] = resourceManager->newObject<RayQueue>(_maxQueueSize, resourceManager);
            _missQueue = resourceManager->newObject<MissQueue>(_maxQueueSize, resourceManager);
            _mediaEvaQueue = resourceManager->newObject<MediaEvaQueue>(_maxQueueSize, resourceManager);
            _materialEvaQueue = resourceManager->newObject<MaterialEvaQueue>(_maxQueueSize, resourceManager);
            _areaLightEvaQueue = resourceManager->newObject<AreaLightHitQueue>(_maxQueueSize, resourceManager);
            _shadowRayQueue = resourceManager->newObject<ShadowRayQueue>(_maxQueueSize, resourceManager);

            _pixelArray = resourceManager->newObject<PixelStateArray>(resourceManager);
            _pixelArray->reset(_maxQueueSize);

            _film = resourceManager->newObject<Film>(params.filmWidth, params.filmHeight, resourceManager);
            _lights = parsedScene.lights;
            _envLights = parsedScene.envLights;
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
                        _mediaEvaQueue->reset();
                        _materialEvaQueue->reset();
                        _areaLightEvaQueue->reset();
                        _shadowRayQueue->reset();
                        nextRayQueue(bounce)->reset();

                        generateRaySamples(sampleIndex, bounce);
                        _world->findClosestHit(currentRayQueue(bounce), _missQueue, _materialEvaQueue,
                                               _mediaEvaQueue, _areaLightEvaQueue, _pixelArray);
                        evaluateMediumSample(bounce);
                        evaluateMissRays();
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
            auto func = RENDER_CPU_GPU_LAMBDA(int
            idx) {
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

        void PathTracer::evaluateMediumSample(int bounce) {
            RayQueue *nextQueue = nextRayQueue(bounce);
            auto func = RENDER_CPU_GPU_LAMBDA(MediaEvaDetails & m) {
                if (!m.medium.nullable()) {
                    MediumInteraction mi;
                    RNG rng = RNG(hash(m.ray.getStep()), hash(m.ray.getDirection()));
                    m.ray.setStep(m.tMax);

                    PixelState &state = (*_pixelArray)[m.pixelIndex];
                    Spectrum Tr = m.medium.sample(m.ray, state.raySamples.media.mediaDist, rng, &mi);
                    state.beta *= Tr;

                    if (mi.isValid()) {
                        Vector3F wi;
                        // TODO delete pdf ?
                        mi.phaseFunc.sample(m.si.wo, &wi, state.raySamples.media.scatter);

                        RayDetails r;
                        r.pixelIndex = m.bounce + 1;
                        r.specularBounce = false;
                        r.ray = mi.generateRay(wi);
                        nextQueue->enqueue(r);

                        // Sample from light
                        Vector3F lightDirection;
                        Float samplePdf, lightPdf;
                        samplePdf = lightPdf = 0.f;
                        Interaction target;
                        bool isDeltaType = false;
                        Spectrum Li = sampleDirectLight(_lights, mi, state.raySamples.sampleLight, lightDirection,
                                                        lightPdf, samplePdf, target, isDeltaType);

                        if (!Li.isBlack() && samplePdf > 0.f) {
                            ShadowRayDetails details;
                            details.ray = mi.generateRayTo(target);
                            if (details.ray.getStep() > details.ray.getMinStep()) {
                                details.tMax = details.ray.getStep();
                                details.beta = state.beta * mi.phaseFunc.pdf(m.si.wo, lightDirection);
                                details.scatterPdf = mi.phaseFunc.pdf(m.si.wo, lightDirection);
                                details.sampleLightPdf = samplePdf;
                                details.deltaType = isDeltaType;
                                details.pixelIndex = m.pixelIndex;
                                details.L = Li;
                                details.lightPdf = lightPdf;

                                RNG rng = RNG(hash(details.ray.getStep()), hash(details.ray.getDirection()));
                                details.Tr = m.medium.transmittance(details.ray, rng);

                                _shadowRayQueue->enqueue(details);
                            }
                        }
                        return;
                    }
                }

                if (m.areaLight != nullptr) {
                    AreaLightHitDetails areaLightHitDetails;
                    areaLightHitDetails.areaLight = m.areaLight;
                    areaLightHitDetails.pixelIndex = m.pixelIndex;
                    areaLightHitDetails.bounce = m.bounce;
                    areaLightHitDetails.si = m.si;
                    areaLightHitDetails.specularBounce = m.specularBounce;
                    _areaLightEvaQueue->enqueue(areaLightHitDetails);
                }

                MaterialEvaDetails materialEvaDetails;
                materialEvaDetails.bounce = m.bounce;
                materialEvaDetails.pixelIndex = m.pixelIndex;
                materialEvaDetails.si = m.si;
                materialEvaDetails.material = m.material;
                _materialEvaQueue->enqueue(materialEvaDetails);
            };
            parallel::parallelForQueue(func, _mediaEvaQueue, _maxQueueSize);
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
                // Sample light
                if (bsdf.allIncludeOf(BxDFType(BSDF_All & (~BSDF_Specular)))) {
                    Vector3F lightDirection;
                    Float samplePdf = 0;
                    Interaction target;
                    bool isDeltaType = false;
                    Float lightPdf = 0.f;
                    Spectrum Li = sampleDirectLight(_lights, m.si, state.raySamples.sampleLight, lightDirection,
                                                    lightPdf, samplePdf, target, isDeltaType);
                    if (!Li.isBlack() && samplePdf > 0.) {
                        Float cosTheta = ABS_DOT(m.si.ns, lightDirection);
                        ShadowRayDetails details;

                        details.ray = m.si.generateRayTo(target);
                        if (details.ray.getStep() > details.ray.getMinStep()) {
                            details.tMax = details.ray.getStep();
                            details.beta = state.beta * bsdf.f(m.si.wo, lightDirection) * cosTheta;
                            details.scatterPdf = bsdf.samplePdf(m.si.wo, lightDirection);
                            details.sampleLightPdf = samplePdf;
                            details.deltaType = isDeltaType;
                            details.pixelIndex = m.pixelIndex;
                            details.L = Li;
                            details.lightPdf = lightPdf;

                            if (details.ray.getMedium().nullable()) {
                                details.Tr = Spectrum(1.0f);
                            } else {
                                RNG rng = RNG(hash(details.ray.getStep()), hash(details.ray.getDirection()));
                                details.Tr = details.ray.getMedium().transmittance(details.ray, rng);
                            }

                            _shadowRayQueue->enqueue(details);
                        }
                    }
                }

                Vector3F wi;
                Float pdf;
                BxDFType sampleType;
                // sample direction
                Spectrum f = bsdf.sampleF(m.si.wo, &wi, &pdf, state.raySamples.scatter, &sampleType);
                if (f.isBlack() || pdf == 0.) {
                    return;
                }
                // evaluate beta
                Float cosTheta = ABS_DOT(m.si.ns, NORMALIZE(wi));
                state.beta *= (f * cosTheta / pdf);

                Spectrum rrBeta = state.beta * (1.0f / bsdf.refractionIndex);
                if (rrBeta.maxComponent() < 1.0f && m.bounce > 15) {
                    Float q = std::max(0.0f, 1 - rrBeta.maxComponent());
                    if (state.raySamples.rr < q) {
                        return;
                    }
                }

                RayDetails r;
                r.pixelIndex = m.pixelIndex;
                r.bounce = m.bounce + 1;
                r.specularBounce = (sampleType & BSDF_Specular) > 0;
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
                // 4 for CameraSamples + 3: scatter + 3: sampleLight + 3: mediumSample + 1: rr
                int dimension = 4 + (3 + 3 + 3 + 1) * bounce;

                PixelState &pixelState = (*_pixelArray)[pixelIndex];

                SamplerType sampler = (*_sampler.cast<SamplerType>());
                sampler.setCurrentSample({pixelState.pixelX, pixelState.pixelY}, sampleIndex, dimension);

                pixelState.raySamples.scatter.u = sampler.sample1D();
                pixelState.raySamples.scatter.uv = sampler.sample2D();

                pixelState.raySamples.sampleLight.light = sampler.sample1D();
                pixelState.raySamples.sampleLight.uv = sampler.sample2D();

                pixelState.raySamples.media.mediaDist = sampler.sample1D();
                pixelState.raySamples.media.scatter = sampler.sample2D();

                pixelState.raySamples.rr = sampler.sample1D();
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
            auto func = RENDER_CPU_GPU_LAMBDA(int
            idx) {
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

        void PathTracer::evaluateMissRays() {
            auto func = RENDER_CPU_GPU_LAMBDA(RayDetails & r) {
                PixelState &state = (*_pixelArray)[r.pixelIndex];
                if (r.bounce == 0 || r.specularBounce) {
                    if (_envLights != nullptr) {
                        Spectrum Le(0);
                        for (int i = 0; i < _envLights->size(); i++) {
                            Le += (*_envLights)[i]->Le(r.ray);
                        }
                        state.L += state.beta * Le;
                    }
                }
            };
            parallel::parallelForQueue(func, _missQueue, _maxQueueSize);
        }

        RayQueue *PathTracer::currentRayQueue(int depth) {
            return _rayQueues[depth % 2];
        }

        RayQueue *PathTracer::nextRayQueue(int depth) {
            return _rayQueues[(depth + 1) % 2];
        }
    }
}