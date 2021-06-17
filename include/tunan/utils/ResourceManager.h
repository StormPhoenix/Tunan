//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_RESOURCEMANAGER_H
#define TUNAN_RESOURCEMANAGER_H

#include <tunan/common.h>
#include <tunan/utils/memory/Allocator.h>

#include <list>
#include <cstddef>
#include <cstdint>

namespace RENDER_NAMESPACE {
    namespace utils {
        class ResourceManager {
        public:
            ResourceManager();

            ResourceManager(Allocator *allocator);

            ResourceManager(const ResourceManager &allocator) = delete;

//            ResourceManager(const ResourceManager &allocator) :
//                    _allocator(allocator._allocator), _defaultBlockSize(1024 * 1024),
//                    _currentBlock(nullptr), _blockOffset(0), _allocatedBlockSize(0) {}

            void *allocate(size_t bytes, size_t alignBytes);

            template<class T>
            T *allocateObjects(size_t count = 1, size_t alignBytes = sizeof(T)) {
                return static_cast<T *>(allocate(sizeof(T) * count, alignBytes));
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

            ResourceManager &operator=(const ResourceManager &allocator) = delete;

            void deleteObject(void *p);

            void reset();

            ~ResourceManager();

            const Allocator *getResource() const {
                return _allocator;
            }

        private:
            Allocator *_allocator;
            size_t _defaultBlockSize;

            uint8_t *_currentBlock = nullptr;
            size_t _blockOffset;
            size_t _allocatedBlockSize;

            std::list<std::pair<size_t, uint8_t *>> _usedBlocks;
            std::list<std::pair<size_t, uint8_t *>> _availableBlocks;
        };

        inline bool operator==(const ResourceManager &alloc1, const ResourceManager &alloc2) noexcept {
            return alloc1.getResource() == alloc2.getResource();
        }
    }
}

#endif //TUNAN_RESOURCEMANAGER_H
