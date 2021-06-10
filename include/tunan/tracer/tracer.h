//
// Created by StormPhoenix on 2021/6/6.
//

#ifndef TUNAN_TRACER_H
#define TUNAN_TRACER_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/scene/Ray.h>
#include <tunan/scene/lights.h>
#include <tunan/material/material.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/containers.h>
#include <tunan/base/interactions.h>

namespace RENDER_NAMESPACE {
    namespace tracer {
        using namespace base;
        using namespace material;

        typedef struct RayDetails {
            Ray ray;
            int pixelIndex;
            int bounce = 0;
            bool specularBounce = false;
        } RayDetails;

        typedef struct MaterialEvaDetails {
            int bounce;
            int pixelIndex;
            Material material;
            SurfaceInteraction si;
        } MaterialEvaDetails;

        typedef struct MediaEvaDetails {
            // TODO
        } MediaEvaDetails;

        typedef struct RaySamples {
            // 3 samples
            struct {
                Float u;
                Point2F uv;
            } scatter;

            struct {
                Float light;
                Point2F uv;
            } sampleLight;
        } RaySamples;

        typedef struct PixelState {
            int pixelX, pixelY;
            RaySamples raySamples;
            Spectrum L;
            Spectrum beta;
        } PixelState;

        typedef struct AreaLightHitDetails {
            int bounce;
            bool specularBounce;
            int pixelIndex;
            DiffuseAreaLight *areaLight = nullptr;
            SurfaceInteraction si;
        } AreaLightHitDetails;

        typedef struct ShadowRayDetails {
            Float lightPdf;
            Float scatterPdf;
            Float sampleLightPdf;
            bool deltaType = false;
            int pixelIndex;
            Spectrum L, beta;
            Ray ray;
            float tMax;
        } ShadowRayDetails;

        typedef base::Queue <RayDetails> RayQueue;
        typedef base::Queue <RayDetails> MissQueue;
        typedef base::Queue <ShadowRayDetails> ShadowRayQueue;
        typedef base::Queue <MaterialEvaDetails> MaterialEvaQueue;
        typedef base::Queue <MediaEvaDetails> MediaEvaQueue;
        typedef base::Queue <AreaLightHitDetails> AreaLightHitQueue;
        typedef base::Vector <PixelState> PixelStateArray;
    }
}

#endif //TUNAN_TRACER_H
