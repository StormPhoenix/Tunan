//
// Created by StormPhoenix on 2021/6/6.
//

#ifndef TUNAN_TRACER_H
#define TUNAN_TRACER_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/scene/Ray.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/containers.h>

namespace RENDER_NAMESPACE {
    namespace tracer {
        using base::Spectrum;

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
    }
}

#endif //TUNAN_TRACER_H
