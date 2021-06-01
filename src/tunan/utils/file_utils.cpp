//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/utils/file_utils.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        bool extensionEquals(const std::string &filename, const std::string &extension) {
            if (extension.size() > filename.size()) return false;
            return std::equal(
                    extension.rbegin(), extension.rend(), filename.rbegin(),
                    [](char a, char b) { return std::tolower(a) == std::tolower(b); });
        }
    }
}