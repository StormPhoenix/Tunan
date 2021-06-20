//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MEDIUMS_H
#define TUNAN_MEDIUMS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/scene/Ray.h>
#include <tunan/sampler/rng.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/medium/phase_functions.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class MediumInteraction;
    }
    using namespace base;
    using namespace sampler;
    using utils::TaggedPointer;

    class HomogenousMedium {
    public:
        HomogenousMedium(const Spectrum &sigma_a, const Spectrum &sigma_s, Float g);

        RENDER_CPU_GPU
        Spectrum transmittance(const Ray &ray, Float u, RNG rng) const;

        RENDER_CPU_GPU
        Spectrum sample(const Ray &ray, Float u, RNG rng, MediumInteraction *mi) const;

    private:
        Spectrum _sigma_a, _sigma_s;
        Spectrum _sigma_t;
        Float _g;
        HGFunc _phaseFunc;
    };

    class Medium : public TaggedPointer<HomogenousMedium> {
    public:
        using TaggedPointer::TaggedPointer;
    };

    class MediumInterface {
    public:
        RENDER_CPU_GPU
        MediumInterface() {}

        RENDER_CPU_GPU
        MediumInterface(Medium m) : inside(m), outside(m) {}

        RENDER_CPU_GPU
        MediumInterface(Medium m1, Medium m2) : inside(m1), outside(m2) {}

        Medium inside, outside;
    };
}

#endif //TUNAN_MEDIUMS_H
