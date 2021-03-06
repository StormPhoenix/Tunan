//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_SAMPLERS_H
#define TUNAN_SAMPLERS_H

#include <tunan/common.h>
#include <tunan/math.h>
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

        class Sampler : public utils::TaggedPointer<IndependentSampler> {
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

        RENDER_CPU_GPU
        inline Vector2F diskUniformSampling(const Point2F &uv, Float radius = 1.) {
            // sampleY = r / Radius
            // sampleX = theta / (2 * PI)
            Float sampleY = uv[0];
            Float sampleX = uv[1];

            Float theta = 2 * Pi * sampleX;
            Float r = sampleY * radius;

            return Vector2F(r * std::cos(theta), r * std::sin(theta));
        }

        RENDER_CPU_GPU
        inline Vector3F hemiCosineSampling(const Vector2F &uv) {
            Vector2F sample = uv;
            // fi = 2 * Pi * sampleU
            Float sampleU = sample.x;
            // sampleV = sin^2(theta)
            Float sampleV = sample.y;
            // x = sin(theta) * cos(fi)
            Float x = std::sqrt(sampleV) * std::cos(2 * Pi * sampleU);
            // y = cos(theta)
            Float y = std::sqrt(1 - sampleV);
            // z = sin(theta) * sin(fi)
            Float z = std::sqrt(sampleV) * std::sin(2 * Pi * sampleU);
            return NORMALIZE(Vector3F(x, y, z));
        }

        RENDER_CPU_GPU
        inline Vector2F triangleUniformSampling(Vector2F uv) {
            Float u = 1 - std::sqrt(uv[0]);
            Float v = uv[1] * std::sqrt(uv[0]);
            return Vector2F(u, v);
        }

        RENDER_CPU_GPU
        inline Float uniformDouble(Float min, Float max, Float sample) {
            return min + (max - min) * sample;
        }

        RENDER_CPU_GPU
        inline int uniformInteger(int min, int max, Float sample) {
            int ret = std::min(static_cast<int>(uniformDouble(min, max + 1, sample)), max);
            if (ret <= max) {
                return ret;
            } else {
                return uniformInteger(min, max, sample);
            }
        }
    }
}

#endif //TUNAN_SAMPLERS_H
