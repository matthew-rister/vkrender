module;

#include <utility>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module image;

import allocator;

namespace vktf {

export class Image {
public:
  Image(const vk::Device device,
        const vk::Format format,
        const vk::Extent2D extent,
        const std::uint32_t mip_levels,
        const vk::SampleCountFlagBits sample_count,
        const vk::ImageUsageFlags usage_flags,
        const vk::ImageAspectFlagBits aspect_mask,
        const VmaAllocator allocator,
        const VmaAllocationCreateInfo& allocation_create_info = kDefaultAllocationCreateInfo);

  Image(const Image&) = delete;
  Image(Image&& image) noexcept { *this = std::move(image); }

  Image& operator=(const Image&) = delete;
  Image& operator=(Image&& image) noexcept;

  ~Image() noexcept;

  [[nodiscard]] vk::ImageView image_view() const noexcept { return *image_view_; }
  [[nodiscard]] vk::Format format() const noexcept { return format_; }

  void Copy(const vk::Buffer buffer,
            const std::vector<vk::BufferImageCopy>& buffer_image_copies,
            const vk::CommandBuffer command_buffer) const;

private:
  vk::Image image_;
  vk::UniqueImageView image_view_;
  vk::Format format_ = vk::Format::eUndefined;
  std::uint32_t mip_levels_ = 0;
  vk::ImageAspectFlagBits aspect_mask_ = vk::ImageAspectFlagBits::eNone;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

}  // namespace vktf

module :private;

namespace {

void TransitionImageLayout(const vk::Image image,
                           const vk::ImageSubresourceRange& image_subresource_range,
                           const std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags>& stage_masks,
                           const std::pair<vk::AccessFlags, vk::AccessFlags>& access_masks,
                           const std::pair<vk::ImageLayout, vk::ImageLayout>& image_layouts,
                           const vk::CommandBuffer command_buffer) {
  const auto& [src_stage_mask, dst_stage_mask] = stage_masks;
  const auto& [src_access_mask, dst_access_mask] = access_masks;
  const auto& [old_layout, new_layout] = image_layouts;

  command_buffer.pipelineBarrier(src_stage_mask,
                                 dst_stage_mask,
                                 vk::DependencyFlags{},
                                 nullptr,
                                 nullptr,
                                 vk::ImageMemoryBarrier{.srcAccessMask = src_access_mask,
                                                        .dstAccessMask = dst_access_mask,
                                                        .oldLayout = old_layout,
                                                        .newLayout = new_layout,
                                                        .image = image,
                                                        .subresourceRange = image_subresource_range});
}

}  // namespace

namespace vktf {

Image::Image(const vk::Device device,
             const vk::Format format,
             const vk::Extent2D extent,
             const std::uint32_t mip_levels,
             const vk::SampleCountFlagBits sample_count,
             const vk::ImageUsageFlags usage_flags,
             const vk::ImageAspectFlagBits aspect_mask,
             const VmaAllocator allocator,
             const VmaAllocationCreateInfo& allocation_create_info)
    : format_{format}, mip_levels_{mip_levels}, aspect_mask_{aspect_mask}, allocator_{allocator} {
  const VkImageCreateInfo image_create_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = static_cast<VkFormat>(format),
      .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
      .mipLevels = mip_levels_,
      .arrayLayers = 1,
      .samples = static_cast<VkSampleCountFlagBits>(sample_count),
      .usage = static_cast<VkImageUsageFlags>(usage_flags)};

  VkImage image = nullptr;
  const auto result =
      vmaCreateImage(allocator_, &image_create_info, &allocation_create_info, &image, &allocation_, nullptr);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Image creation failed");
  image_ = vk::Image{image};

  image_view_ = device.createImageViewUnique(vk::ImageViewCreateInfo{
      .image = image_,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange =
          vk::ImageSubresourceRange{.aspectMask = aspect_mask_, .levelCount = mip_levels, .layerCount = 1}});
}

Image& Image::operator=(Image&& image) noexcept {
  if (this != &image) {
    image_ = std::exchange(image.image_, nullptr);
    image_view_ = std::exchange(image.image_view_, vk::UniqueImageView{});
    format_ = std::exchange(image.format_, vk::Format::eUndefined);
    allocator_ = std::exchange(image.allocator_, nullptr);
    allocation_ = std::exchange(image.allocation_, nullptr);
  }
  return *this;
}

Image::~Image() noexcept {
  if (allocator_ != nullptr) {
    vmaDestroyImage(allocator_, image_, allocation_);
  }
}

void Image::Copy(const vk::Buffer buffer,
                 const std::vector<vk::BufferImageCopy>& buffer_image_copies,
                 const vk::CommandBuffer command_buffer) const {
  const vk::ImageSubresourceRange image_subresource_range{.aspectMask = aspect_mask_,
                                                          .levelCount = mip_levels_,
                                                          .layerCount = 1};

  TransitionImageLayout(image_,
                        image_subresource_range,
                        std::pair{vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer},
                        std::pair{vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite},
                        std::pair{vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal},
                        command_buffer);

  command_buffer.copyBufferToImage(buffer, image_, vk::ImageLayout::eTransferDstOptimal, buffer_image_copies);

  TransitionImageLayout(image_,
                        image_subresource_range,
                        std::pair{vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader},
                        std::pair{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead},
                        std::pair{vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
                        command_buffer);
}

}  // namespace vktf
