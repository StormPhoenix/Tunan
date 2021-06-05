//
// Created by StormPhoenix on 2021/6/1.
//

#ifndef TUNAN_IMAGE_UTILS_H
#define TUNAN_IMAGE_UTILS_H

#include <tunan/common.h>
#include <string>

namespace RENDER_NAMESPACE {
    namespace utils {
        float *readImage(const std::string &filename, int *width, int *height,
                         int desire_channels, int *channels_in_file);

        void writeImage(const std::string filename, int width, int height, int channel,
                        const unsigned char *input_image);
    }
}

#endif //TUNAN_IMAGE_UTILS_H
