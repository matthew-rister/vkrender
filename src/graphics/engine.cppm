module;

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module engine;

import allocator;
import buffer;
import camera;
import delta_time;
import device;
import image;
import instance;
import physical_device;
import scene;
import swapchain;
import window;

namespace gfx {

export class Engine {
public:
  explicit Engine(const Window& window);

  [[nodiscard]] Scene Load(const std::filesystem::path& asset_filepath) const;

  void Run(const Window& window, std::invocable<DeltaTime> auto&& main_loop_fn) const {
    for (DeltaTime delta_time; !window.ShouldClose();) {
      delta_time.Update();
      window.Update();
      main_loop_fn(delta_time);
    }
    device_->waitIdle();
  }

  void Render(const Scene& scene, const Camera& camera);

private:
  static constexpr std::size_t kMaxRenderFrames = 2;
  std::size_t current_frame_index_ = 0;
  Instance instance_;
  vk::UniqueSurfaceKHR surface_;
  PhysicalDevice physical_device_;
  Device device_;
  Allocator allocator_;
  Swapchain swapchain_;
  vk::SampleCountFlagBits msaa_sample_count_ = vk::SampleCountFlagBits::e1;
  Image color_attachment_;
  Image depth_attachment_;
  vk::UniqueRenderPass render_pass_;
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  vk::Queue graphics_queue_;
  vk::Queue present_queue_;
  vk::UniqueCommandPool draw_command_pool_;
  std::vector<vk::UniqueCommandBuffer> draw_command_buffers_;
  std::array<vk::UniqueFence, kMaxRenderFrames> draw_fences_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> acquire_next_image_semaphores_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> present_image_semaphores_;
};

}  // namespace gfx

module :private;

namespace {

vk::Extent2D GetFrameBufferExtent(const gfx::Window& window) {
  const auto& [width, height] = window.GetSize();
  return vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)};
}

vk::SampleCountFlagBits GetMsaaSampleCount(const vk::PhysicalDeviceLimits& physical_device_limits) {
  using enum vk::SampleCountFlagBits;
  static constexpr auto kTargetSampleCountFlagBits = {e8, e4, e2, e1};

  const auto color_sample_count_flags = physical_device_limits.framebufferColorSampleCounts;
  const auto depth_sample_count_flags = physical_device_limits.framebufferDepthSampleCounts;
  const auto color_depth_sample_count_flags = color_sample_count_flags & depth_sample_count_flags;

  const auto iterator =
      std::ranges::find_if(kTargetSampleCountFlagBits,
                           [color_depth_sample_count_flags](const auto sample_count_flag_bit) {
                             return static_cast<bool>(sample_count_flag_bit & color_depth_sample_count_flags);
                           });

  assert(iterator != std::cend(kTargetSampleCountFlagBits));  // at least one sample count is required
  return *iterator;
}

vk::Format GetDepthAttachmentFormat(const vk::PhysicalDevice physical_device) {
  using enum vk::Format;
  static constexpr auto kTargetDepthAttachmentFormats = {eD32Sfloat, eX8D24UnormPack32, eD16Unorm};

  const auto iterator = std::ranges::find_if(kTargetDepthAttachmentFormats, [physical_device](const auto format) {
    const auto optimal_tiling_features = physical_device.getFormatProperties(format).optimalTilingFeatures;
    return static_cast<bool>(optimal_tiling_features & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  });

  // the Vulkan specification requires VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT support for VK_FORMAT_D16_UNORM
  // and at least one of VK_FORMAT_X8_D24_UNORM_PACK32 and VK_FORMAT_D32_SFLOAT
  assert(iterator != std::cend(kTargetDepthAttachmentFormats));
  return *iterator;
}

vk::UniqueRenderPass CreateRenderPass(const vk::Device device,
                                      const vk::SampleCountFlagBits msaa_sample_count,
                                      const vk::Format color_attachment_format,
                                      const vk::Format depth_attachment_format) {
  const vk::AttachmentDescription color_attachment_description{.format = color_attachment_format,
                                                               .samples = msaa_sample_count,
                                                               .loadOp = vk::AttachmentLoadOp::eClear,
                                                               .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                               .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                               .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                               .initialLayout = vk::ImageLayout::eUndefined,
                                                               .finalLayout = vk::ImageLayout::eColorAttachmentOptimal};

  const vk::AttachmentDescription color_resolve_attachment_description{
      .format = color_attachment_format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR};

  const vk::AttachmentDescription depth_attachment_description{
      .format = depth_attachment_format,
      .samples = msaa_sample_count,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

  const std::array attachment_descriptions{color_attachment_description,
                                           color_resolve_attachment_description,
                                           depth_attachment_description};

  static constexpr vk::AttachmentReference kColorAttachmentReference{
      .attachment = 0,
      .layout = vk::ImageLayout::eColorAttachmentOptimal};

  static constexpr vk::AttachmentReference kColorResolveAttachmentReference{
      .attachment = 1,
      .layout = vk::ImageLayout::eColorAttachmentOptimal};

  static constexpr vk::AttachmentReference kDepthAttachmentReference{
      .attachment = 2,
      .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

  static constexpr vk::SubpassDescription kSubpassDescription{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                                              .colorAttachmentCount = 1,
                                                              .pColorAttachments = &kColorAttachmentReference,
                                                              .pResolveAttachments = &kColorResolveAttachmentReference,
                                                              .pDepthStencilAttachment = &kDepthAttachmentReference};

  static constexpr vk::SubpassDependency kSubpassDependency{
      .srcSubpass = vk::SubpassExternal,
      .dstSubpass = 0,
      .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .srcAccessMask = vk::AccessFlagBits::eNone,
      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite};

  return device.createRenderPassUnique(
      vk::RenderPassCreateInfo{.attachmentCount = static_cast<std::uint32_t>(attachment_descriptions.size()),
                               .pAttachments = attachment_descriptions.data(),
                               .subpassCount = 1,
                               .pSubpasses = &kSubpassDescription,
                               .dependencyCount = 1,
                               .pDependencies = &kSubpassDependency});
}

std::vector<vk::UniqueFramebuffer> CreateFramebuffers(const vk::Device device,
                                                      const gfx::Swapchain& swapchain,
                                                      const vk::RenderPass render_pass,
                                                      const vk::ImageView color_attachment,
                                                      const vk::ImageView depth_attachment) {
  return swapchain.image_views()
         | std::views::transform([=, extent = swapchain.image_extent()](const auto color_resolve_attachment) {
             const std::array image_attachments{color_attachment, color_resolve_attachment, depth_attachment};
             return device.createFramebufferUnique(
                 vk::FramebufferCreateInfo{.renderPass = render_pass,
                                           .attachmentCount = static_cast<std::uint32_t>(image_attachments.size()),
                                           .pAttachments = image_attachments.data(),
                                           .width = extent.width,
                                           .height = extent.height,
                                           .layers = 1});
           })
         | std::ranges::to<std::vector>();
}

template <std::size_t MaxRenderFrames>
std::array<vk::UniqueFence, MaxRenderFrames> CreateFences(const vk::Device device) {
  std::array<vk::UniqueFence, MaxRenderFrames> fences;
  std::ranges::generate(fences, [device] {
    // create the fence in a signaled state to avoid waiting on the first frame
    return device.createFenceUnique(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  });
  return fences;
}

template <std::size_t MaxRenderFrames>
std::array<vk::UniqueSemaphore, MaxRenderFrames> CreateSemaphores(const vk::Device device) {
  std::array<vk::UniqueSemaphore, MaxRenderFrames> semaphores;
  std::ranges::generate(semaphores, [device] { return device.createSemaphoreUnique(vk::SemaphoreCreateInfo{}); });
  return semaphores;
}

}  // namespace

namespace gfx {

Engine::Engine(const Window& window)
    : surface_{window.CreateSurface(*instance_)},
      physical_device_{*instance_, *surface_},
      device_{physical_device_},
      allocator_{*instance_, *physical_device_, *device_},
      swapchain_{*surface_,
                 *physical_device_,
                 *device_,
                 GetFrameBufferExtent(window),
                 physical_device_.queue_family_indices()},
      msaa_sample_count_{GetMsaaSampleCount(physical_device_.limits())},
      color_attachment_{*device_,
                        swapchain_.image_format(),
                        swapchain_.image_extent(),
                        1,
                        msaa_sample_count_,
                        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
                        vk::ImageAspectFlagBits::eColor,
                        *allocator_,
                        VmaAllocationCreateInfo{.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                                .usage = VMA_MEMORY_USAGE_AUTO,
                                                .priority = 1.0f}},
      depth_attachment_{*device_,
                        GetDepthAttachmentFormat(*physical_device_),
                        swapchain_.image_extent(),
                        1,
                        msaa_sample_count_,
                        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
                        vk::ImageAspectFlagBits::eDepth,
                        *allocator_,
                        VmaAllocationCreateInfo{.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                                .usage = VMA_MEMORY_USAGE_AUTO,
                                                .priority = 1.0f}},
      render_pass_{
          CreateRenderPass(*device_, msaa_sample_count_, swapchain_.image_format(), depth_attachment_.format())},
      framebuffers_{CreateFramebuffers(*device_,
                                       swapchain_,
                                       *render_pass_,
                                       color_attachment_.image_view(),
                                       depth_attachment_.image_view())},
      graphics_queue_{device_->getQueue(physical_device_.queue_family_indices().graphics_index, 0)},
      present_queue_{device_->getQueue(physical_device_.queue_family_indices().present_index, 0)},
      draw_command_pool_{device_->createCommandPoolUnique(
          vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                    .queueFamilyIndex = physical_device_.queue_family_indices().graphics_index})},
      draw_command_buffers_{
          device_->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *draw_command_pool_,
                                                                              .level = vk::CommandBufferLevel::ePrimary,
                                                                              .commandBufferCount = kMaxRenderFrames})},
      draw_fences_{CreateFences<kMaxRenderFrames>(*device_)},
      acquire_next_image_semaphores_{CreateSemaphores<kMaxRenderFrames>(*device_)},
      present_image_semaphores_{CreateSemaphores<kMaxRenderFrames>(*device_)} {}

Scene Engine::Load(const std::filesystem::path& asset_filepath) const {
  // TODO(matthew-rister): add support for loading .glb files
  if (const auto extension = asset_filepath.extension(); extension != ".gltf") {
    throw std::runtime_error{std::format("Unsupported file extension: {}", extension.string())};
  }
  return Scene{
      asset_filepath,
      SubmitCopyCommandsOptions{.device = *device_,
                                // TODO(matthew-rister): use a dedicated transfer queue to copy assets to device memory
                                .transfer_queue = graphics_queue_,
                                .transfer_queue_family_index = physical_device_.queue_family_indices().graphics_index,
                                .allocator = *allocator_},
      CreateTextureOptions{.physical_device = *physical_device_,
                           .enable_sampler_anisotropy = physical_device_.features().samplerAnisotropy,
                           .max_sampler_anisotropy = physical_device_.limits().maxSamplerAnisotropy},
      CreateGraphicsPipelineOptions{.viewport_extent = swapchain_.image_extent(),
                                    .msaa_sample_count = msaa_sample_count_,
                                    .render_pass = *render_pass_}};
}

void Engine::Render(const Scene& scene, const Camera& camera) {
  if (++current_frame_index_ == kMaxRenderFrames) {
    current_frame_index_ = 0;
  }

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index): index guaranteed to be within array bounds
  const auto draw_fence = *draw_fences_[current_frame_index_];
  const auto acquire_next_image_semaphore = *acquire_next_image_semaphores_[current_frame_index_];
  const auto present_image_semaphore = *present_image_semaphores_[current_frame_index_];
  // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  auto result = device_->waitForFences(draw_fence, vk::True, kMaxTimeout);
  vk::detail::resultCheck(result, "Draw fence failed to enter a signaled state");
  device_->resetFences(draw_fence);

  std::uint32_t image_index = 0;
  std::tie(result, image_index) = device_->acquireNextImageKHR(*swapchain_, kMaxTimeout, acquire_next_image_semaphore);
  vk::detail::resultCheck(result, "Acquire next swapchain image failed");

  static constexpr std::array kClearColor{0.0f, 0.0f, 0.0f, 1.0f};
  static constexpr std::array kClearValues{vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
                                           vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
                                           vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};

  const auto draw_command_buffer = *draw_command_buffers_[current_frame_index_];
  draw_command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  draw_command_buffer.beginRenderPass(
      vk::RenderPassBeginInfo{
          .renderPass = *render_pass_,
          .framebuffer = *framebuffers_[image_index],
          .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = swapchain_.image_extent()},
          .clearValueCount = static_cast<std::uint32_t>(kClearValues.size()),
          .pClearValues = kClearValues.data()},
      vk::SubpassContents::eInline);

  scene.Render(camera, draw_command_buffer);

  draw_command_buffer.endRenderPass();
  draw_command_buffer.end();

  static constexpr vk::PipelineStageFlags kPipelineWaitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  graphics_queue_.submit(vk::SubmitInfo{.waitSemaphoreCount = 1,
                                        .pWaitSemaphores = &acquire_next_image_semaphore,
                                        .pWaitDstStageMask = &kPipelineWaitStage,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &draw_command_buffer,
                                        .signalSemaphoreCount = 1,
                                        .pSignalSemaphores = &present_image_semaphore},
                         draw_fence);

  const auto swapchain = *swapchain_;
  result = present_queue_.presentKHR(vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                                                        .pWaitSemaphores = &present_image_semaphore,
                                                        .swapchainCount = 1,
                                                        .pSwapchains = &swapchain,
                                                        .pImageIndices = &image_index});
  vk::detail::resultCheck(result, "Present swapchain image failed");
}

}  // namespace gfx