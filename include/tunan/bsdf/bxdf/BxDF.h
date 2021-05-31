//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_BXDF_H
#define TUNAN_BXDF_H

#include <tunan/common.h>

namespace RENDER_NAMESPACE {
    namespace bsdf {
        typedef enum {
            IMPORTANCE,
            RADIANCE
        } TransportMode;
    }
}

#endif //TUNAN_BXDF_H
