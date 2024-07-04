#ifndef GRAPHICS_BUFFER_H_
#define GRAPHICS_BUFFER_H_

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "graphics/data_view.h"

namespace gfx {

class Buffer {
public:
  Buffer(vk::DeviceSize size_bytes,
         vk::BufferUsageFlags usage_flags,
         VmaAllocator allocator,
         const VmaAllocationCreateInfo& allocation_create_info);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  ~Buffer() noexcept;

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  template <typename T>
  void Copy(const DataView<const T> data_view) {
    assert(data_view.size_bytes() <= size_bytes_);
    auto* mapped_memory = MapMemory();
    memcpy(mapped_memory, data_view.data(), size_bytes_);
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

  template <typename T>
  void CopyOnce(const DataView<const T> data_view) {
    Copy(data_view);
    UnmapMemory();
  }

private:
  void* MapMemory();
  void UnmapMemory() noexcept;

  vk::Buffer buffer_;
  vk::DeviceSize size_bytes_ = 0;
  void* mapped_memory_ = nullptr;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

}  // namespace gfx

#endif  // GRAPHICS_BUFFER_H_