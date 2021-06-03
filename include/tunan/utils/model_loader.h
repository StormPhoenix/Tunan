//
// Created by StormPhoenix on 2021/6/3.
//

#ifndef TUNAN_MODEL_LOADER_H
#define TUNAN_MODEL_LOADER_H

#include <tunan/common.h>
#include <string>
#include <vector>

namespace RENDER_NAMESPACE {
    class ShapeEntity;
    namespace utils {
        bool load_obj(const std::string model_path, ShapeEntity &entity);
    }
}

#endif //TUNAN_MODEL_LOADER_H
