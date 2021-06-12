//
// Created by Storm Phoenix on 2021/6/11.
//

#ifndef TUNAN_MICROFACETS_H
#define TUNAN_MICROFACETS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    namespace microfacet {
        using utils::TaggedPointer;

        class GGXDistribution {
        public:
            RENDER_CPU_GPU
            GGXDistribution(Float alpha_g);

            RENDER_CPU_GPU
            Float D(const Normal3F &wh) const;

            RENDER_CPU_GPU
            Float G(const Vector3F &v, const Normal3F &wh) const;

            RENDER_CPU_GPU
            Float G(const Vector3F &wo, const Vector3F &wi, const Normal3F &wh) const;

            RENDER_CPU_GPU
            Vector3F sampleWh(const Vector3F &wo, Vector2F &uv) const;

        private:
            // Width parameter g
            Float _alpha_g;
        };

        class MicrofacetDistribution : public TaggedPointer<GGXDistribution> {
        public:
            using TaggedPointer::TaggedPointer;
            // Normal distribution function D(wh)
            RENDER_CPU_GPU
            Float D(const Normal3F &wh) const;

            // Mask and shadowing function G(wi, wh)
            // G(wi, wh) is independent of wh, so rewrite G(wi, wh) to G(wi)
            // Wh direction is positive hemi-sphere by default
            RENDER_CPU_GPU
            Float G(const Vector3F &v, const Normal3F &wh) const;

            // Mask and shadowing function G(wo, wi)
            // Which gives the fraction of microfacets in a differential area that are visible from
            //  both directions w_o and w_i
            RENDER_CPU_GPU
            Float G(const Vector3F &wo, const Vector3F &wi, const Normal3F &wh) const;

            // Sample wh
            RENDER_CPU_GPU
            Vector3F sampleWh(const Vector3F &wo, Vector2F &uv) const;

            // Sample wh pdf
            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &wo, const Vector3F &wh) const;
        };
    }
}

#endif //TUNAN_MICROFACETS_H
