//
// Created by Storm Phoenix on 2021/6/18.
//

#include <tunan/base/hash.h>
#include <tunan/base/interactions.h>
#include <tunan/medium/mediums.h>
#include <tunan/sampler/samplers.h>

namespace RENDER_NAMESPACE {
    RENDER_CPU_GPU
    HomogenousMedium::HomogenousMedium(const Spectrum &sigma_a, const Spectrum &sigma_s, Float g) :
            _sigma_a(sigma_a), _sigma_s(sigma_s), _sigma_t(sigma_a + sigma_s), _g(g) {
        _phaseFunc = HGFunc(_g);
    }

    RENDER_CPU_GPU
    Spectrum HomogenousMedium::transmittance(const Ray &ray, Float u, RNG rng) const {
        return exp(-_sigma_t * std::min(ray.getStep() * LENGTH(ray.getDirection()), MaxFloat));
    }

    RENDER_CPU_GPU
    Spectrum HomogenousMedium::sample(const Ray &ray, Float u, RNG rng, MediumInteraction *mi) const {
        // different channel has different sigma, randomly chose a channel
        int channel = sampler::uniformInteger(0, SpectrumChannel - 1, rng.uniform<Float>());

        // sample1d Tr uniformly, and calculate the correspond dist
        Float dist = -std::log(1 - u) / _sigma_t[channel];
        Float step = dist / LENGTH(ray.getDirection());

        // check whether sample1d the surface or medium
        bool sampleMedium = step < ray.getStep();
        if (sampleMedium) {
            (*mi) = MediumInteraction(ray.at(step), -ray.getDirection(), this, &_phaseFunc);
        } else {
            step = ray.getStep();
        }

        // calculate transmittance
        Spectrum T = exp(-_sigma_t * std::min(step, MaxFloat) * LENGTH(ray.getDirection()));

        // calculate pdf
        Spectrum p = sampleMedium ? _sigma_t * T : T;
        Float pdf = 0;
        for (int i = 0; i < SpectrumChannel; i++) {
            pdf += p[i];
        }
        pdf /= SpectrumChannel;
        if (pdf == 0) {
            assert(T.isBlack());
        }
        return sampleMedium ? (T * _sigma_s / pdf) : (T / pdf);
    }
}