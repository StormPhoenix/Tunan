//
// Created by StormPhoenix on 2021/6/16.
//

#ifndef TUNAN_DISTRIBUTIONS_H
#define TUNAN_DISTRIBUTIONS_H

#include <tunan/common.h>
#include <tunan/math.h>
#include <tunan/base/containers.h>
#include <tunan/utils/ResourceManager.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class Distribution1D {
        public:
            Distribution1D() = default;

            Distribution1D(Float *func, int len, ResourceManager *allocator);

            RENDER_CPU_GPU
            Float sampleContinuous(Float *pdf, int *offset, Float sample) const;

            RENDER_CPU_GPU
            Float pdfDiscrete(int index) const;

            RENDER_CPU_GPU
            int length() const;

            friend class Distribution2D;

        private:
            Float _funcIntegration;
            Vector<Float> _function;
            Vector<Float> _cdf;
        };

        class Distribution2D {
        public:
            Distribution2D() = default;

            Distribution2D(Float *func2D, int width, int height, ResourceManager *allocator);

            RENDER_CPU_GPU
            Point2F sampleContinuous(Float *pdf, Vector2F &uv) const;

            RENDER_CPU_GPU
            Float pdfContinuous(const Point2F &uv) const;

        private:
            Vector<Distribution1D> _rowDistribution;
            Distribution1D _marginalDistribution;
            int _width, _height;
        };
    }
}

#endif //TUNAN_DISTRIBUTIONS_H
