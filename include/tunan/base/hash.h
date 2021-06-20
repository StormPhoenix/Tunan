//
// Created by Storm Phoenix on 2021/6/19.
//

#ifndef TUNAN_HASH_H
#define TUNAN_HASH_H

#include <tunan/common.h>
#include <inttypes.h>
#include <cstddef>
#include <string>

namespace RENDER_NAMESPACE {
    namespace base {
        template<typename... Args>
        RENDER_CPU_GPU
        static inline void copyArgs(char *buffer, Args... args);

        template<>
        RENDER_CPU_GPU
        static inline void copyArgs(char *buffer) {}

        template<typename T, typename... Args>
        RENDER_CPU_GPU
        static inline void copyArgs(char *buffer, T v, Args... args) {
            memcpy(buffer, &v, sizeof(T));
            copyArgs(buffer + sizeof(T), args...);
        }

        RENDER_CPU_GPU
        static inline uint64_t MurmurHash64A(const void *key, int len, uint64_t seed) {
            const uint64_t m = 0xc6a4a7935bd1e995ull;
            const int r = 47;

            uint64_t h = seed ^(len * m);

            const uint64_t *data = (const uint64_t *) key;
            const uint64_t *end = data + (len / 8);

            while (data != end) {
                uint64_t k = *data++;

                k *= m;
                k ^= k >> r;
                k *= m;

                h ^= k;
                h *= m;
            }

            const unsigned char *data2 = (const unsigned char *) data;

            switch (len & 7) {
                case 7:
                    h ^= uint64_t(data2[6]) << 48;
                case 6:
                    h ^= uint64_t(data2[5]) << 40;
                case 5:
                    h ^= uint64_t(data2[4]) << 32;
                case 4:
                    h ^= uint64_t(data2[3]) << 24;
                case 3:
                    h ^= uint64_t(data2[2]) << 16;
                case 2:
                    h ^= uint64_t(data2[1]) << 8;
                case 1:
                    h ^= uint64_t(data2[0]);
                    h *= m;
            };

            h ^= h >> r;
            h *= m;
            h ^= h >> r;

            return h;
        }

        template<typename... Args>
        RENDER_CPU_GPU
        inline uint64_t hash(Args... args) {
            // Reference: https://stackoverflow.com/questions/57246592
            constexpr size_t argSize = (sizeof(Args) + ... + 0);
            constexpr size_t longByteSize = (argSize + 7) / 8;
            uint64_t buffer[longByteSize];
            copyArgs((char *) buffer, args...);
            return MurmurHash64A(buffer, argSize, 0);
        }
    }
}

#endif //TUNAN_HASH_H
