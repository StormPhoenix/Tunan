//
// Created by StormPhoenix on 2021/6/16.
//

#include <tunan/math.h>
#include <tunan/base/distributions.h>

#include <vector>

namespace RENDER_NAMESPACE {
    namespace base {
        template<typename Judge>
        RENDER_CPU_GPU
        int binarySearch(int size, const Judge &judge) {
            int start = 0;
            int len = size;
            while (len > 0) {
                int half = len >> 1;
                int middle = start + half;
                if (judge(middle)) {
                    start = middle + 1;
                    len -= (half + 1);
                } else {
                    len = half;
                }
            }
            return math::clamp(start - 1, 0, size - 2);
        }

        Distribution1D::Distribution1D(Float *func, int len, MemoryAllocator &allocator)
                : _function(allocator), _cdf(allocator) {
//                : _function(func, func + len), _cdf(len + 1) {
            _function.reset(len);
            _cdf.reset(len + 1);
            _cdf[0] = 0.;
            Float delta = 1.0 / len;
            for (int i = 1; i <= len; i++) {
                _cdf[i] = _cdf[i - 1] + delta * _function[i - 1];
            }

            // Normalization
            _funcIntegration = _cdf[len];
            if (_funcIntegration == 0.) {
                for (int i = 0; i <= len; i++) {
                    _cdf[i] = Float(i) / Float(len);
                }
            } else {
                for (int i = 1; i <= len; i++) {
                    _cdf[i] /= _funcIntegration;
                }
            }
        }

        RENDER_CPU_GPU
        Float Distribution1D::sampleContinuous(Float *pdf, int *offset, Float sample) const {
            Float u = sample;
            int discreteOffset = binarySearch(_cdf.size(), [&](int index) -> bool { return _cdf[index] <= u; });
            if (offset != nullptr) {
                (*offset) = discreteOffset;
            }

            if (pdf != nullptr) {
                (*pdf) = _funcIntegration > 0 ? _function[discreteOffset] / _funcIntegration : 0.;
            }

            Float dSample = u - _cdf[discreteOffset];
            return (dSample / (_cdf[discreteOffset + 1] - _cdf[discreteOffset]) + discreteOffset) / length();
        }

        RENDER_CPU_GPU
        Float Distribution1D::pdfDiscrete(int index) const {
            if (index < 0 || index >= length()) {
                return 0.;
            }
            Float delta = 1.0 / length();
            return _funcIntegration > 0 ? _function[index] / _funcIntegration * delta : 0.;
        }

        RENDER_CPU_GPU
        int Distribution1D::length() const {
            return _function.size();
        }

        Distribution2D::Distribution2D(Float *func2D, int width, int height, MemoryAllocator &allocator) :
                _width(width), _height(height), _rowDistribution(allocator) {
            _rowDistribution.reset(height);
            for (int row = 0; row < height; row++) {
                int offset = row * width;
                _rowDistribution[row] = Distribution1D(func2D + offset, width, allocator);
            }

            std::vector<Float> marginals;
            for (int i = 0; i < height; i++) {
                marginals.push_back(_rowDistribution[i]._funcIntegration);
            }
            _marginalDistribution = Distribution1D(&marginals[0], height, allocator);
        }

        RENDER_CPU_GPU
        Point2F Distribution2D::sampleContinuous(Float *pdf, Vector2F &uv) const {
            Float marginalPdf;
            int marginalDiscreteOffset;
            Float marginalOffset = _marginalDistribution.sampleContinuous(&marginalPdf, &marginalDiscreteOffset,
                                                                          uv[0]);

            Float colPdf;
            int colDiscreteOffset;
            Float colOffset = _rowDistribution[marginalDiscreteOffset].sampleContinuous(&colPdf,
                                                                                         &colDiscreteOffset,
                                                                                         uv[1]);

            if (pdf != nullptr) {
                (*pdf) = marginalPdf * colPdf;
            }

            return Point2F(colOffset, marginalOffset);
        }

        RENDER_CPU_GPU
        Float Distribution2D::pdfContinuous(const Point2F &uv) const {
            int v = uv[1] * _height;
            int u = uv[0] * _width;

            u = math::clamp(u, 0, _width - 1);
            v = math::clamp(v, 0, _height - 1);

            return _rowDistribution[v]._function[u] / _marginalDistribution._funcIntegration;
        }
    }
}
