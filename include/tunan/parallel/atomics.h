//
// Created by StormPhoenix on 2021/6/9.
//

#ifndef TUNAN_ATOMICS_H
#define TUNAN_ATOMICS_H

#include <tunan/math.h>
#include <tunan/common.h>

#ifndef RENDER_GPU_CODE
#include <atomic>
#endif

namespace RENDER_NAMESPACE {
    namespace parallel {
        class AtomicFloat {
        public:
            RENDER_CPU_GPU
            explicit AtomicFloat(Float val = 0) {
#ifdef RENDER_GPU_CODE
                value = val;
#else
                bits = math::float2bits(val);
#endif
            }

            RENDER_CPU_GPU
            Float get() {
#ifdef RENDER_GPU_CODE
                return value;
#else
                return math::bits2float(bits);
#endif
            }

            RENDER_CPU_GPU
            operator Float() const {
#ifdef RENDER_GPU_CODE
                return value;
#else
                return math::bits2float(bits);
#endif
            }

            RENDER_CPU_GPU
            Float operator=(Float val) {
#ifdef RENDER_GPU_CODE
                value = val;
                return value;
#else
                bits = math::float2bits(val);
                return val;
#endif
            }

            RENDER_CPU_GPU
            void add(float val) {
#ifdef RENDER_GPU_CODE
                atomicAdd(&value, val);
#else
                uint32_t oldBits = bits;
                uint32_t newBits;

                do {
                    newBits = math::float2bits(math::bits2float(oldBits) + val);
                } while (!bits.compare_exchange_weak(oldBits, newBits));
#endif
            }

        private:
#ifdef RENDER_GPU_CODE
            float value;
#else
            std::atomic<uint32_t> bits;
#endif
        };
    }
}

#endif //TUNAN_ATOMICS_H
