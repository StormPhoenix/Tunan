//
// Created by StormPhoenix on 2021/6/1.
//

#ifndef TUNAN_FILE_UTILS_H
#define TUNAN_FILE_UTILS_H

#include <tunan/common.h>
#include <string>

namespace RENDER_NAMESPACE {
    namespace utils {
        bool extensionEquals(const std::string &filename, const std::string &extension);
    }
}

#endif //TUNAN_FILE_UTILS_H
