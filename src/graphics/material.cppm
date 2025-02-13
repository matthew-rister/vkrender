module;

#include <vulkan/vulkan.hpp>

export module material;

import buffer;
import texture;

namespace vktf {

export struct Material {
  Texture base_color_texture;
  Texture metallic_roughness_texture;
  Texture normal_texture;
  Buffer properties_buffer;
  vk::DescriptorSet descriptor_set;
  vk::PipelineLayout pipeline_layout;
};

}  // namespace vktf
