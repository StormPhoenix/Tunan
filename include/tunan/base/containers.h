//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_CONTAINER_H
#define TUNAN_CONTAINER_H

#include <tunan/utils/MemoryAllocator.h>
#include <tunan/common.h>

#ifdef __RENDER_GPU_MODE__
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <initializer_list>
#include <cassert>
#include <atomic>

namespace RENDER_NAMESPACE {
    namespace base {
        using utils::MemoryAllocator;

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
                for (int i = 0; i < N; i++) {
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

        template<typename T>
        class Vector {
        public:
            Vector(const MemoryAllocator &allocator) :
                    allocator(allocator), nAllocated(0), used(0) {}

            RENDER_CPU_GPU
            const T &operator[](size_t index) const {
                assert(index >= 0 && index < used);
                return buffer[index];
            }

            RENDER_CPU_GPU
            T &operator[](size_t index) {
                assert(index >= 0 && index < used);
                return buffer[index];
            }

            RENDER_CPU_GPU
            T *data() {
                return buffer;
            }

            RENDER_CPU_GPU
            const T *data() const {
                return buffer;
            }

            RENDER_CPU_GPU
            size_t size() {
                return used;
            }

            void push_back(T &val) {
                if (used == nAllocated) {
                    allocate(nAllocated == 0 ? 4 : 2 * nAllocated);
                }
                allocator.template initialize<T>(buffer + used, val);
                used++;
            }

            void push_back(T &&val) {
                if (used == nAllocated) {
                    allocate(nAllocated == 0 ? 4 : 2 * nAllocated);
                }
                allocator.template initialize<T>(buffer + used, std::move(val));
                used++;
            }

            void pop_back() {
                if (used <= 0) {
                    return;
                }
                allocator.template de_initialize<T>(buffer + used - 1);
                used--;
            }

            void reset(size_t n) {
                if (n > nAllocated) {
                    allocate(n);
                }
                used = n;
            }

        private:
            void allocate(size_t n) {
                if (n <= nAllocated) {
                    return;
                }

                T *newBuffer = allocator.template allocateObjects<T>(n);
                for (int i = 0; i < used; i++) {
                    allocator.template initialize<T>(newBuffer + i, std::move(buffer[i]));
                }
                buffer = newBuffer;
                nAllocated = n;
            }

        private:
            int nAllocated;
            int used;
            T *buffer = nullptr;
            MemoryAllocator allocator;
        };

        template<typename T>
        class Queue {
        public:
            Queue() = default;

            RENDER_CPU_GPU
            Queue(int maxQueueSize, const MemoryAllocator &allocator) :
                    maxQueueSize(maxQueueSize), allocator(allocator) {
                buffer = allocator.allocateObjects<T>(maxQueueSize);
                nAllocated = maxQueueSize;
            }

            RENDER_CPU_GPU
            T &operator[](size_t index) {
                assert(index >= 0 && index < _size);
                return buffer[index];
            }

            RENDER_CPU_GPU
            const T &operator[](size_t index) const {
                assert(index >= 0 && index < _size);
                return buffer[index];
            }


            RENDER_CPU_GPU
            int enqueue(T &val) {
                int index = push();
                buffer[index] = val;
                return index;
            }

            RENDER_CPU_GPU
            int enqueue(T &&val) {
                int index = push();
                buffer[index] = val;
                return index;
            }

            RENDER_CPU_GPU
            T dequeue() {
                int index = pop() - 1;
                return buffer[index];
            }

            RENDER_CPU_GPU
            int size() const {
#ifdef __RENDER_GPU_MODE__
                return _size;
#else
                return _size.load(std::memory_order_relaxed);
#endif
            }

            RENDER_CPU_GPU
            void reset() {
#ifdef __RENDER_GPU_MODE__
                _size = 0;
#else
                _size.store(0, std::memory_order_relaxed);
#endif
            }

        private:
            int push() {
                assert(_size < maxQueueSize);
#ifdef __RENDER_GPU_MODE__
                return atomicAdd(&_size, 1);
#else
                return _size.fetch_add(1, std::memory_order_relaxed);
#endif
            }

            int pop() {
                assert(_size > 0);
#ifdef __RENDER_GPU_MODE__
                return atomicSub(&_size, 1);
#else
                return _size.fetch_sub(1, std::memory_order_relaxed);
#endif
            }

        private:
            int maxQueueSize = 0;

#ifdef __RENDER_GPU_MODE__
            int _size = 0;
#else
            std::atomic<int> _size{0};
#endif

            int nAllocated = 0;
            T *buffer = nullptr;
            MemoryAllocator allocator;
        };
    }
}

#endif //TUNAN_CONTAINER_H
