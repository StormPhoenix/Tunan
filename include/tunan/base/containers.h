//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_CONTAINER_H
#define TUNAN_CONTAINER_H

#include <cassert>
#include <tunan/common.h>
#include <initializer_list>

namespace RENDER_NAMESPACE {
    namespace base {
        template<typename T, int N>
        class Array {
        public:
            Array() {}

            RENDER_CPU_GPU
            Array(std::initializer_list<T> arr) {
                size_t i = 0;
                for (const T &val : arr) {
                    values[i] = val;
                    i++;
                }
            }

            RENDER_CPU_GPU
            bool operator==(const Array<T, N> &arr) const {
                for (int i = 0; i < N; i++) {
                    if (values[i] != arr.values[i]) {
                        return false;
                    }
                }
                return true;
            }

            RENDER_CPU_GPU
            bool operator!=(const Array<T, N> &arr) const {
                return !((*this) == arr);
            }

            RENDER_CPU_GPU
            T &operator[](size_t i) {
                return values[i];
            }

            RENDER_CPU_GPU
            const T &operator[](size_t i) const {
                return values[i];
            }

            RENDER_CPU_GPU
            const T *begin() const {
                return values;
            }

            RENDER_CPU_GPU
            const T *end() const {
                return values + N;
            }

            RENDER_CPU_GPU
            T *begin() {
                return values;
            }

            RENDER_CPU_GPU
            T *end() {
                return values + N;
            }

            RENDER_CPU_GPU
            void fill(const T &val) {
                for (int i = 0; i < N; i ++) {
                    values[i] = val;
                }
            }

            RENDER_CPU_GPU
            size_t size() {
                return N;
            }

            RENDER_CPU_GPU
            T *data() {
                return values;
            }

        private:
            T values[N];
        };

        template<typename T>
        class Array<T, 0> {
        public:
            Array() {}

            RENDER_CPU_GPU
            bool operator==(const Array<T, 0> &arr) const {
                return true;
            }

            RENDER_CPU_GPU
            bool operator!=(const Array<T, 0> &arr) const {
                return false;
            }

            RENDER_CPU_GPU
            T &operator[](size_t i) {
                // Can not be called
                assert(false);
                T val;
                return val;
            }

            RENDER_CPU_GPU
            const T &operator[](size_t i) const {
                // Can not be called
                assert(false);
                T val;
                return val;
            }

            RENDER_CPU_GPU
            const T *begin() const {
                return nullptr;
            }

            RENDER_CPU_GPU
            const T *end() const {
                return nullptr;
            }

            RENDER_CPU_GPU
            T *begin() {
                return nullptr;
            }

            RENDER_CPU_GPU
            T *end() {
                return nullptr;
            }

            RENDER_CPU_GPU
            size_t size() {
                return 0;
            }

            RENDER_CPU_GPU
            T *data() {
                return nullptr;
            }
        };


    }
}

#endif //TUNAN_CONTAINER_H
