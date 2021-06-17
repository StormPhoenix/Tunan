//
// Created by StormPhoenix on 2021/6/9.
//

#include <tunan/scene/film.h>
#include <tunan/utils/image_utils.h>

namespace RENDER_NAMESPACE {
    Film::Film(int filmWidth, int filmHeight, ResourceManager &allocator) :
            _filmWidth(filmWidth), _filmHeight(filmHeight) {
        int filmSize = _filmWidth * _filmHeight;
        _filmPlane = allocator.allocateObjects<Pixel>(filmSize);
    }

    RENDER_CPU_GPU
    void Film::addExtra(const Spectrum &spectrum, int row, int col) {
        if (row < 0 || row >= _filmHeight ||
            col < 0 || col >= _filmWidth) {
            return;
        }

        int offset = (row * _filmWidth + col);
        Pixel &pixel = _filmPlane[offset];
        for (int ch = 0; ch < SpectrumChannel; ch++) {
            pixel.extra[ch].add(spectrum[ch]);
        }
    }

    RENDER_CPU_GPU
    void Film::addSpectrum(const Spectrum &spectrum, int row, int col) {
        if (row < 0 || row >= _filmHeight ||
            col < 0 || col >= _filmWidth) {
            return;
        }

        int offset = (row * _filmWidth + col);
        Pixel &pixel = _filmPlane[offset];
        pixel.spectrum += spectrum;
    }

    RENDER_CPU_GPU
    void Film::setSpectrum(const Spectrum &spectrum, int row, int col) {
        if (row < 0 || row >= _filmHeight ||
            col < 0 || col >= _filmWidth) {
            return;
        }

        int offset = (row * _filmWidth + col);
        Pixel &pixel = _filmPlane[offset];
        pixel.spectrum = spectrum;
    }

    void Film::writeImage(char const *filename, Float weight) {
        // create image buffer
        unsigned char *image = new unsigned char[_filmWidth * _filmHeight * SpectrumChannel];
        {
            for (int row = _filmHeight - 1; row >= 0; row--) {
                for (int col = 0; col < _filmWidth; col++) {
                    int imageOffset = (row * _filmWidth + col) * SpectrumChannel;
                    int pixelOffset = row * _filmWidth + col;
                    Pixel &pixel = _filmPlane[pixelOffset];
                    for (int ch = 0; ch < SpectrumChannel; ch++) {
                        (image + imageOffset)[ch] = static_cast<unsigned char>(
                                math::clamp(std::sqrt((pixel.spectrum[ch] + pixel.extra[ch].get()) * weight), 0.0,
                                            0.999) * 256 );
                    }
                }
            }
        }
        // write image file
        utils::writeImage(filename, _filmWidth, _filmHeight, SpectrumChannel, image);
        delete[] image;
    }
}