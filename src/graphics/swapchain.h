#ifndef SRC_GRAPHICS_SWAPCHAIN_H_
#define SRC_GRAPHICS_SWAPCHAIN_H_

#include <algorithm>
#include <ranges>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace gfx {
class Device;
class Window;

class Swapchain {
public:
  Swapchain(const Device& device, const Window& window, const vk::SurfaceKHR& surface);

  [[nodiscard]] const vk::SwapchainKHR& operator*() const noexcept { return *swapchain_; }
  [[nodiscard]] const vk::SwapchainKHR* operator->() const noexcept { return &(*swapchain_); }

  [[nodiscard]] vk::Format image_format() const noexcept { return image_format_; }
  [[nodiscard]] const vk::Extent2D& image_extent() const noexcept { return image_extent_; }

  [[nodiscard]] std::ranges::constant_range auto image_views() const {
    return image_views_ | std::views::transform([](const auto& image_view) -> auto& { return *image_view; });
  }

private:
  vk::UniqueSwapchainKHR swapchain_;
  vk::Format image_format_{};
  vk::Extent2D image_extent_;
  std::vector<vk::UniqueImageView> image_views_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_SWAPCHAIN_H_
