module;

#include <array>
#include <cassert>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <vulkan/vulkan.hpp>

export module mesh;

import buffer;
import data_view;
import material;

namespace vktf {

export struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};
  glm::vec2 texture_coordinates_0{0.0f};
};

struct IndexBuffer {
  Buffer buffer;
  vk::IndexType index_type = vk::IndexType::eUint16;
  std::uint32_t index_count = 0;
};

export class Primitive {
public:
  template <typename IndexType>
    requires std::is_same_v<IndexType, std::uint16_t> || std::is_same_v<IndexType, std::uint32_t>
  Primitive(std::vector<Vertex> vertices, std::vector<IndexType> indices, const Material* material)
      : vertices_{std::move(vertices)}, indices_{std::move(indices)}, material_{material} {}

  // returns vertex and index staging buffers which must remain in scope until command buffer submission completes
  std::array<Buffer, 2> CreateBuffers(const vk::CommandBuffer command_buffer, const VmaAllocator allocator);

  void Render(const vk::CommandBuffer command_buffer, const vk::PipelineLayout graphics_pipeline_layout) const;

private:
  using VertexData = std::vector<Vertex>;
  using IndexData = std::variant<std::vector<std::uint16_t>, std::vector<std::uint32_t>>;

  std::variant<VertexData, Buffer> vertices_;
  std::variant<IndexData, IndexBuffer> indices_;
  const Material* material_ = nullptr;
};

export using Mesh = std::vector<Primitive>;

}  // namespace vktf

module :private;

namespace {

template <typename T>
std::array<vktf::Buffer, 2> CreateDeviceLocalBuffer(const vktf::DataView<const T> data_view,
                                                    const vk::BufferUsageFlags usage_flags,
                                                    const vk::CommandBuffer command_buffer,
                                                    const VmaAllocator allocator) {
  auto host_buffer = vktf::CreateHostVisibleBuffer(data_view, allocator);
  vktf::Buffer device_buffer{data_view.size_bytes(), usage_flags | vk::BufferUsageFlagBits::eTransferDst, allocator};
  command_buffer.copyBuffer(*host_buffer, *device_buffer, vk::BufferCopy{.size = data_view.size_bytes()});
  return std::array{std::move(host_buffer), std::move(device_buffer)};
}

}  // namespace

namespace vktf {

std::array<Buffer, 2> Primitive::CreateBuffers(const vk::CommandBuffer command_buffer, const VmaAllocator allocator) {
  const auto* const vertex_data = std::get_if<VertexData>(&vertices_);
  assert(vertex_data != nullptr);
  auto [vertex_staging_buffer, vertex_buffer] =
      CreateDeviceLocalBuffer<VertexData>(*vertex_data,
                                          vk::BufferUsageFlagBits::eVertexBuffer,
                                          command_buffer,
                                          allocator);
  vertices_ = std::move(vertex_buffer);

  const auto index_data_visitor = [command_buffer, allocator]<typename T>(const T& index_data_t) {
    if constexpr (std::is_same_v<std::uint16_t, typename T::value_type>) {
      auto [index_staging_buffer, index_buffer] =
          CreateDeviceLocalBuffer<std::uint16_t>(index_data_t,
                                                 vk::BufferUsageFlagBits::eIndexBuffer,
                                                 command_buffer,
                                                 allocator);

      return std::pair{std::move(index_staging_buffer),
                       IndexBuffer{.buffer = std::move(index_buffer),
                                   .index_type = vk::IndexType::eUint16,
                                   .index_count = static_cast<std::uint32_t>(index_data_t.size())}};
    } else {
      static_assert(std::is_same_v<std::uint32_t, typename T::value_type>);
      auto [index_staging_buffer, index_buffer] =
          CreateDeviceLocalBuffer<std::uint32_t>(index_data_t,
                                                 vk::BufferUsageFlagBits::eIndexBuffer,
                                                 command_buffer,
                                                 allocator);
      return std::pair{std::move(index_staging_buffer),
                       IndexBuffer{.buffer = std::move(index_buffer),
                                   .index_type = vk::IndexType::eUint32,
                                   .index_count = static_cast<std::uint32_t>(index_data_t.size())}};
    }
  };

  const auto* const index_data = std::get_if<IndexData>(&indices_);
  assert(index_data != nullptr);

  auto [index_staging_buffer, index_buffer] = std::visit(index_data_visitor, *index_data);
  indices_ = std::move(index_buffer);

  return std::array{std::move(vertex_staging_buffer), std::move(index_staging_buffer)};
}

void Primitive::Render(const vk::CommandBuffer command_buffer,
                       const vk::PipelineLayout graphics_pipeline_layout) const {
  // TODO: avoid per-primitive descriptor set binding
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                    graphics_pipeline_layout,
                                    1,
                                    material_->descriptor_set,
                                    nullptr);

  const auto* const vertex_buffer = std::get_if<Buffer>(&vertices_);
  assert(vertex_buffer != nullptr);
  command_buffer.bindVertexBuffers(0, **vertex_buffer, static_cast<vk::DeviceSize>(0));

  const auto* const index_buffer = std::get_if<IndexBuffer>(&indices_);
  assert(index_buffer != nullptr);
  command_buffer.bindIndexBuffer(*index_buffer->buffer, 0, index_buffer->index_type);
  command_buffer.drawIndexed(index_buffer->index_count, 1, 0, 0, 0);
}

}  // namespace vktf
