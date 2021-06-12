//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_SAMPLERS_H
#define TUNAN_SAMPLERS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/sampler/rng.h>

namespace RENDER_NAMESPACE {
    namespace sampler {
        class IndependentSampler {
        public:
            IndependentSampler(int nSamples, int seed = 0) :
                    _nSamples(nSamples), _seed(seed) {}

            RENDER_CPU_GPU
            void setCurrentSample(const Point2I pixel, int sampleIndex, int dimension);

            RENDER_CPU_GPU
            void forPixel(const Point2I pixel);

            RENDER_CPU_GPU
            void setSampleIndex(int sampleIndex);

            RENDER_CPU_GPU
            void advance(int dimension);

            RENDER_CPU_GPU
            bool nextSampleRound();

            RENDER_CPU_GPU
            Float sample1D();

            RENDER_CPU_GPU
            Point2F sample2D();

        private:
            // Current pixel on which sampling
            Point2I _currentPixel;
            // Sample times
            const int _nSamples;
            // Random number seed
            int _seed = 0;
            // Sample index
            int _sampleIndex = 0;

            RNG rng;
        };

        using utils::TaggedPointer;

        class Sampler : public TaggedPointer<IndependentSampler> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU inline void setCurrentSample(const Point2I pixel, int sampleIndex, int dimension);

            RENDER_CPU_GPU inline void forPixel(const Point2I pixel);

            RENDER_CPU_GPU inline void setSampleIndex(int sampleIndex);

            RENDER_CPU_GPU inline void advance(int dimension);

            RENDER_CPU_GPU inline bool nextSampleRound();

            RENDER_CPU_GPU inline Float sample1D();

            RENDER_CPU_GPU inline Vector2F sample2D();
        };

        RENDER_CPU_GPU Vector2F diskUniformSampling(const Point2F &uv, Float radius = 1.);

        RENDER_CPU_GPU Vector3F hemiCosineSampling(const Vector2F &uv);

        RENDER_CPU_GPU
        inline Vector2F triangleUniformSampling(Vector2F uv) {
            Float u = 1 - std::sqrt(uv[0]);
            Float v = uv[1] * sqrt(uv[0]);
            return Vector2F(u, v);
        }
    }
}

#endif //TUNAN_SAMPLERS_H
