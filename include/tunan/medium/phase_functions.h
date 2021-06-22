//
// Created by Storm Phoenix on 2021/6/18.
//

#ifndef TUNAN_PHASE_FUNCTIONS_H
#define TUNAN_PHASE_FUNCTIONS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    using utils::TaggedPointer;

    class HGFunc {
        // henyey greenstein
    public:
        RENDER_CPU_GPU
        HGFunc() = default;

        RENDER_CPU_GPU
        HGFunc(Float g);

        RENDER_CPU_GPU
        Float sample(const Vector3F &wo, Vector3F *wi, Vector2F sample) const;

        RENDER_CPU_GPU
        Float pdf(const Vector3F &wo, const Vector3F &wi) const;

    private:
        Float _G = 0.f;
    };

    class PhaseFunction : public TaggedPointer<HGFunc> {
    public:
        using TaggedPointer::TaggedPointer;

        RENDER_CPU_GPU
        inline Float sample(const Vector3F &wo, Vector3F *wi, Vector2F sample) const {
            auto func = [&](auto ptr) {
                return ptr->sample(wo, wi, sample);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Float pdf(const Vector3F &wo, const Vector3F &wi) const {
            auto func = [&](auto ptr) {
                return ptr->pdf(wo, wi);
            };
            return proxyCall(func);
        }
    };
}

#endif //TUNAN_PHASE_FUNCTIONS_H
