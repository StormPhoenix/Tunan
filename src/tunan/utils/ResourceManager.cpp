//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef RENDER_CACHE_LINE_SIZE
// Cache line size
#define RENDER_CACHE_LINE_SIZE 128
#endif

#include <tunan/utils/ResourceManager.h>
#include <tunan/utils/memory/HostAllocator.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        ResourceManager::ResourceManager()
                : _defaultBlockSize(1024 * 1024), _currentBlock(nullptr),
                  _blockOffset(0), _allocatedBlockSize(0) {
            _allocator = new HostAllocator();
        }

        ResourceManager::ResourceManager(Allocator *allocator)
                : _defaultBlockSize(1024 * 1024), _currentBlock(nullptr),
                  _blockOffset(0), _allocatedBlockSize(0),
                  _allocator(allocator) {}

        void *ResourceManager::allocate(size_t bytes, size_t alignBytes) {
            if ((_blockOffset % alignBytes) != 0) {
                _blockOffset += (alignBytes - (_blockOffset % alignBytes));
            }

            // Aligned
//            bytes = (bytes + alignBytes - 1) & (~(alignBytes - 1));
            if (_blockOffset + bytes > _allocatedBlockSize) {
                if (_currentBlock != nullptr) {
                    _usedBlocks.push_back(std::make_pair(_allocatedBlockSize, _currentBlock));
                    _currentBlock = nullptr;
                    _allocatedBlockSize = 0;
                }

                for (auto iter = _availableBlocks.begin(); iter != _availableBlocks.end(); iter++) {
                    // Search for available size block
                    if (iter->first >= bytes) {
                        _allocatedBlockSize = iter->first;
                        _currentBlock = iter->second;
                        _availableBlocks.erase(iter);
                        break;
                    }
                }

                if (_currentBlock == nullptr) {
                    size_t allocatedBytes = std::max(bytes, _defaultBlockSize);
                    _currentBlock = static_cast<uint8_t *>(
                            _allocator->allocateAlignedMemory(allocatedBytes, RENDER_CACHE_LINE_SIZE));
                    _allocatedBlockSize = allocatedBytes;
                }
                _blockOffset = 0;
            }
            void *ret = _currentBlock + _blockOffset;
            _blockOffset += bytes;
            return ret;
        }

        void ResourceManager::reset() {
            _availableBlocks.splice(_availableBlocks.begin(), _usedBlocks);
            _blockOffset = 0;
        }

        void ResourceManager::deleteObject(void *p) {
            // TODO do nothing
        }

        ResourceManager::~ResourceManager() {
            _allocator->freeAlignedMemory(_currentBlock);
            for (auto iter = _usedBlocks.begin(); iter != _usedBlocks.end(); iter++) {
                _allocator->freeAlignedMemory(iter->second);
            }

            for (auto iter = _availableBlocks.begin(); iter != _availableBlocks.end(); iter++) {
                _allocator->freeAlignedMemory(iter->second);
            }
            // TODO delete
//            delete _resource;
        }
    }
}