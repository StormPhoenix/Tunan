//
// Created by Graphics on 2021/5/31.
//

#ifndef TUNAN_MEDIUMS_H
#define TUNAN_MEDIUMS_H

#include <tunan/common.h>
#include <tunan/medium/Medium.h>

namespace RENDER_NAMESPACE {
    namespace base {
        using medium::Medium;

        class MediumInterface {
        public:
            RENDER_CPU_GPU
            MediumInterface(Medium m) : inside(m), outside(m) {}

            Medium inside, outside;
        };
    }
}

#endif //TUNAN_MEDIUMS_H
