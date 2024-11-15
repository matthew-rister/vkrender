module;

#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include <glslang/Include/glslang_c_interface.h>
#include <vulkan/vulkan.hpp>

export module shader_module;

import glslang_compiler;

namespace gfx {

export class ShaderModule {
public:
  ShaderModule(vk::Device device, vk::ShaderStageFlagBits shader_stage, const std::filesystem::path& glsl_filepath);

  [[nodiscard]] vk::ShaderModule operator*() const noexcept { return *shader_module_; }

private:
  vk::UniqueShaderModule shader_module_;
};

}  // namespace gfx

module :private;

namespace {

glslang_stage_t GetGlslangStage(const vk::ShaderStageFlagBits shader_stage) {
  switch (shader_stage) {  // NOLINT(clang-diagnostic-switch-enum)
    case vk::ShaderStageFlagBits::eVertex:
      return GLSLANG_STAGE_VERTEX;
    case vk::ShaderStageFlagBits::eTessellationControl:
      return GLSLANG_STAGE_TESSCONTROL;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
      return GLSLANG_STAGE_TESSEVALUATION;
    case vk::ShaderStageFlagBits::eGeometry:
      return GLSLANG_STAGE_GEOMETRY;
    case vk::ShaderStageFlagBits::eFragment:
      return GLSLANG_STAGE_FRAGMENT;
    case vk::ShaderStageFlagBits::eCompute:
      return GLSLANG_STAGE_COMPUTE;
    case vk::ShaderStageFlagBits::eRaygenKHR:
      return GLSLANG_STAGE_RAYGEN;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
      return GLSLANG_STAGE_ANYHIT;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
      return GLSLANG_STAGE_CLOSESTHIT;
    case vk::ShaderStageFlagBits::eMissKHR:
      return GLSLANG_STAGE_MISS;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
      return GLSLANG_STAGE_INTERSECT;
    case vk::ShaderStageFlagBits::eCallableKHR:
      return GLSLANG_STAGE_CALLABLE;
    default:
      throw std::runtime_error{std::format("Unsupported shader stage {}", vk::to_string(shader_stage))};
  }
}

std::string ReadFile(const std::filesystem::path& glsl_filepath) {
  if (std::ifstream ifstream{glsl_filepath, std::ios::ate}) {
    const std::streamsize size = ifstream.tellg();
    std::string glsl(static_cast<std::size_t>(size), '\0');
    ifstream.seekg(0, std::ios::beg);
    ifstream.read(glsl.data(), size);
    return glsl;
  }
  throw std::runtime_error{std::format("Failed to open {}", glsl_filepath.string())};
}

}  // namespace

namespace gfx {

ShaderModule::ShaderModule(const vk::Device device,
                           const vk::ShaderStageFlagBits shader_stage,
                           const std::filesystem::path& glsl_filepath) {
  const auto glslang_stage = GetGlslangStage(shader_stage);
  const auto glsl_source = ReadFile(glsl_filepath);
  const auto spirv = glslang::Compile(glslang_stage, glsl_source);

  shader_module_ = device.createShaderModuleUnique(
      vk::ShaderModuleCreateInfo{.codeSize = spirv.size() * sizeof(decltype(spirv)::value_type),
                                 .pCode = spirv.data()});
}

}  // namespace gfx
