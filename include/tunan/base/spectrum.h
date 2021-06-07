//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_SPECTRUM_H
#define TUNAN_SPECTRUM_H

#include <tunan/common.h>
#include <tunan/math.h>
#include <tunan/base/containers.h>

#include <cassert>

namespace RENDER_NAMESPACE {
    namespace base {
        using namespace math;
        static const int SpectrumChannel = 3;

        class Spectrum {
        public:
            RENDER_CPU_GPU
            Spectrum() = default;

            RENDER_CPU_GPU
            explicit Spectrum(Float val) {
                values.fill(val);
            }

            RENDER_CPU_GPU
            Float operator[](int i) const {
                CHECK(i >= 0 && i < SpectrumChannel);
                return values[i];
            }

            RENDER_CPU_GPU
            Float &operator[](int i) {
                CHECK(i >= 0 && i < SpectrumChannel);
                return values[i];
            }

            RENDER_CPU_GPU
            explicit operator bool() const {
                for (int i = 0; i < SpectrumChannel; ++i) {
                    if (values[i] != 0)
                        return true;
                }
                return false;
            }

            RENDER_CPU_GPU
            bool hasNans() const {
                for (int i = 0; i < SpectrumChannel; i++) {
                    if (isNaN(values[i])) {
                        return true;
                    }
                }
                return false;
            }

            RENDER_CPU_GPU
            Spectrum operator+(const Spectrum &v) const {
                Spectrum ret = *this;
                return ret += v;
            }

            RENDER_CPU_GPU
            Spectrum &operator+=(const Spectrum &v) {
                for (int i = 0; i < SpectrumChannel; i++) {
                    values[i] += v.values[i];
                }
                return *this;
            }

            RENDER_CPU_GPU
            bool operator==(const Spectrum &v) const {
                return values == v.values;
            }

            RENDER_CPU_GPU
            bool operator!=(const Spectrum &v) const {
                return values != v.values;
            }

            RENDER_CPU_GPU
            Spectrum &operator-=(const Spectrum &v) {
                for (int i = 0; i < SpectrumChannel; ++i)
                    values[i] -= v.values[i];
                return *this;
            }

            RENDER_CPU_GPU
            Spectrum operator-(const Spectrum &v) const {
                Spectrum ret = *this;
                return ret -= v;
            }

            RENDER_CPU_GPU
            friend Spectrum operator-(Float a, const Spectrum &v) {
                CHECK(!isNaN(a));
                Spectrum ret;
                for (int i = 0; i < SpectrumChannel; ++i)
                    ret.values[i] = a - v.values[i];
                return ret;
            }

            RENDER_CPU_GPU
            Spectrum &operator*=(const Spectrum &v) {
                for (int i = 0; i < SpectrumChannel; ++i)
                    values[i] *= v.values[i];
                return *this;
            }

            RENDER_CPU_GPU
            Spectrum operator*(const Spectrum &v) const {
                Spectrum ret = *this;
                return ret *= v;
            }

            RENDER_CPU_GPU
            Spectrum operator*(Float a) const {
                CHECK(!isNaN(a));
                Spectrum ret = *this;
                for (int i = 0; i < SpectrumChannel; ++i)
                    ret.values[i] *= a;
                return ret;
            }

            RENDER_CPU_GPU
            Spectrum &operator*=(Float a) {
                CHECK(!isNaN(a));
                for (int i = 0; i < SpectrumChannel; ++i)
                    values[i] *= a;
                return *this;
            }

            RENDER_CPU_GPU
            friend Spectrum operator*(Float a, const Spectrum &v) {
                return v * a;
            }

            RENDER_CPU_GPU
            Spectrum &operator/=(const Spectrum &v) {
                for (int i = 0; i < SpectrumChannel; ++i) {
                    CHECK(v.values[i] != 0);
                    values[i] /= v.values[i];
                }
                return *this;
            }

            RENDER_CPU_GPU
            Spectrum operator/(const Spectrum &v) const {
                Spectrum ret = *this;
                return ret /= v;
            }

            RENDER_CPU_GPU
            Spectrum &operator/=(Float a) {
                CHECK(a != 0);
                CHECK(!isNaN(a));
                for (int i = 0; i < SpectrumChannel; ++i)
                    values[i] /= a;
                return *this;
            }

            RENDER_CPU_GPU
            Spectrum operator/(Float a) const {
                Spectrum ret = *this;
                return ret /= a;
            }

            RENDER_CPU_GPU
            Spectrum operator-() const {
                Spectrum ret;
                for (int i = 0; i < SpectrumChannel; ++i)
                    ret.values[i] = -values[i];
                return ret;
            }

        private:
            Array<Float, SpectrumChannel> values;
        };
    }
}

#endif //TUNAN_SPECTRUM_H
