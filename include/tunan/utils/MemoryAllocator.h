//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MEMORYALLOCATOR_H
#define TUNAN_MEMORYALLOCATOR_H

#include <tunan/common.h>
#include <tunan/utils/memory/MemoryResource.h>

#include <list>
#include <cstddef>
#include <cstdint>

namespace RENDER_NAMESPACE {
    namespace utils {
        class MemoryAllocator {
        public:
            MemoryAllocator();

            MemoryAllocator(MemoryResource *resource);

            MemoryAllocator(const MemoryAllocator &allocator) :
                    MemoryAllocator(allocator._resource) {}

            void *allocate(size_t bytes, size_t alignBytes);

            template<class T>
            T *allocateObjects(size_t count = 1) {
                return static_cast<T *>(allocate(sizeof(T) * count, sizeof(T)));
            }

            template<class T, class... Args>
            T *newObject(Args &&... args) {
                T *obj = allocateObjects<T>(1);
                initialize<T>(obj, args...);
                return obj;
            }

            template<class T, class... Args>
            void initialize(void *obj, Args &&... args) {
                new(obj) T(std::forward<Args>(args)...);
            }

            template<class T>
            void de_initialize(T *obj) {
                obj->~T();
            }

            MemoryAllocator &operator=(const MemoryAllocator &allocator) = delete;

            void deleteObject(void *p);

            void reset();

            ~MemoryAllocator();

        private:
            MemoryResource *_resource;
            size_t _defaultBlockSize;

            uint8_t *_currentBlock;
            size_t _blockOffset;
            size_t _allocatedBlockSize;

            std::list<std::pair<size_t, uint8_t *>> _usedBlocks;
            std::list<std::pair<size_t, uint8_t *>> _availableBlocks;
        };
    }
}

#endif //TUNAN_MEMORYALLOCATOR_H
