#include "graphics/texture2d.h"

#include <cstdint>
#include <memory>
#include <string>

#include <stb_image.h>

#include "graphics/device.h"

namespace {

gfx::Image LoadImage(const gfx::Device& device, const vk::Format format, const std::string& filepath) {
  static constexpr auto kRequiredChannels = 4;
  int width{}, height{}, channels{};

  const std::unique_ptr<std::uint8_t, decltype(&stbi_image_free)> data{
      stbi_load(filepath.c_str(), &width, &height, &channels, kRequiredChannels),
      &stbi_image_free};

  if (data == nullptr) {
    throw std::runtime_error{
        std::format("Failed to load the image at {} with error: {}", filepath, stbi_failure_reason())};
  }

  return gfx::CreateDeviceLocalImage(
      device,
      format,
      vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)},
      vk::SampleCountFlagBits::e1,
      vk::ImageUsageFlagBits::eSampled,
      vk::ImageAspectFlagBits::eColor,
      vk::ArrayProxy<const std::uint8_t>{static_cast<std::uint32_t>(width) * height * kRequiredChannels, data.get()});
}

vk::UniqueSampler CreateSampler(const gfx::Device& device) {
  const auto& physical_device = device.physical_device();
  return device->createSamplerUnique(
      vk::SamplerCreateInfo{.magFilter = vk::Filter::eLinear,
                            .minFilter = vk::Filter::eLinear,
                            .addressModeU = vk::SamplerAddressMode::eRepeat,
                            .addressModeV = vk::SamplerAddressMode::eRepeat,
                            .anisotropyEnable = physical_device.features().samplerAnisotropy,
                            .maxAnisotropy = physical_device.limits().maxSamplerAnisotropy});
}

}  // namespace

gfx::Texture2d::Texture2d(const Device& device, const vk::Format format, const std::filesystem::path& filepath)
    : image_{LoadImage(device, format, filepath.string())}, sampler_{CreateSampler(device)} {}