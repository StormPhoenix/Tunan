//
// Created by StormPhoenix on 2021/6/9.
//

#ifndef TUNAN_FILM_H
#define TUNAN_FILM_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/parallel/atomics.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    using namespace base;
    using utils::MemoryAllocator;
    using parallel::AtomicFloat;

    struct Pixel {
        Spectrum spectrum;
        AtomicFloat extra[SpectrumChannel];

        Pixel() {
            spectrum = Spectrum(0);
            for (int i = 0; i < SpectrumChannel; i++) {
                extra[i] = 0;
            }
        }
    };

    class Film {
    public:
        Film(int filmWidth, int filmHeight, MemoryAllocator &allocator);

        RENDER_CPU_GPU
        void addExtra(const Spectrum &spectrum, int row, int col);

        RENDER_CPU_GPU
        void addSpectrum(const Spectrum &spectrum, int row, int col);

        RENDER_CPU_GPU
        void setSpectrum(const Spectrum &spectrum, int row, int col);

        void writeImage(char const *filename, Float weight = 1.0f);

    private:
        int _filmWidth = 0, _filmHeight = 0;
        Pixel *_filmPlane;
    };

}

#endif //TUNAN_FILM_H
