//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/sampler/samplers.h>

namespace RENDER_NAMESPACE {
    namespace sampler {
        RENDER_CPU_GPU
        void IndependentSampler::forPixel(const Point2I pixel) {
            _currentPixel = pixel;
            rng.newSequence((pixel.x + pixel.y * 65536) | (uint64_t(_seed) << 32));
        }

        RENDER_CPU_GPU
        void IndependentSampler::setSampleIndex(int sampleIndex) {
            _sampleIndex = sampleIndex;
            int px = _currentPixel.x;
            int py = _currentPixel.y;
            rng.newSequence((px + py * 65536) | (uint64_t(_seed) << 32));
            rng.advance(_sampleIndex * 65536 + 0);
        }

        RENDER_CPU_GPU
        bool IndependentSampler::nextSampleRound() {
            bool ret = (++_sampleIndex) < _nSamples;
            rng.advance(_sampleIndex * 65536 + 0);
            return ret;
        }

        RENDER_CPU_GPU
        Float IndependentSampler::sample1D() {
            return rng.uniform<Float>();
        }

        RENDER_CPU_GPU
        Point2F IndependentSampler::sample2D() {
            return {rng.uniform<Float>(), rng.uniform<Float>()};
        }

        RENDER_CPU_GPU
        void Sampler::forPixel(const Point2I pixel) {
            auto forPixelFunc = [&](auto ptr) { return ptr->forPixel(pixel); };
            return proxyCall(forPixelFunc);
        }

        RENDER_CPU_GPU
        void Sampler::setSampleIndex(int sampleIndex) {
            auto ssiFunc = [&](auto ptr) { return ptr->setSampleIndex(sampleIndex); };
            return proxyCall(ssiFunc);
        }

        RENDER_CPU_GPU
        bool Sampler::nextSampleRound() {
            auto nsrFunc = [&](auto ptr) { return ptr->nextSampleRound(); };
            return proxyCall(nsrFunc);
        }

        RENDER_CPU_GPU
        Float Sampler::sample1D() {
            auto sample1DFunc = [&](auto ptr) { return ptr->sample1D(); };
            return proxyCall(sample1DFunc);
        }

        RENDER_CPU_GPU
        Vector2F Sampler::sample2D() {
            auto sample2DFunc = [&](auto ptr) { return ptr->sample2D(); };
            return proxyCall(sample2DFunc);
        }

        RENDER_CPU_GPU Vector2F diskUniformSampling(Sampler sampler, Float radius) {
            // sampleY = r / Radius
            // sampleX = theta / (2 * PI)
            Float sampleY = sampler.sample1D();
            Float sampleX = sampler.sample1D();

            Float theta = 2 * Pi * sampleX;
            Float r = sampleY * radius;

            return Vector2F(r * std::cos(theta), r * std::sin(theta));
        }
    }
}