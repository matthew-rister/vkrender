#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace gfx {

class PhysicalDevice {
public:
  struct QueueFamilyIndices {
    std::uint32_t graphics_index{};
    std::uint32_t present_index{};
    std::uint32_t transfer_index{};
  };

  PhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface);

  [[nodiscard]] vk::PhysicalDevice operator*() const noexcept { return physical_device_; }
  [[nodiscard]] const vk::PhysicalDevice* operator->() const noexcept { return &physical_device_; }

  [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return limits_; }
  [[nodiscard]] const vk::PhysicalDeviceMemoryProperties& memory_properties() const noexcept { return mem_properties_; }

  [[nodiscard]] const QueueFamilyIndices& queue_family_indices() const noexcept { return queue_family_indices_; }

private:
  vk::PhysicalDevice physical_device_;
  vk::PhysicalDeviceLimits limits_;
  vk::PhysicalDeviceMemoryProperties mem_properties_;
  QueueFamilyIndices queue_family_indices_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_
