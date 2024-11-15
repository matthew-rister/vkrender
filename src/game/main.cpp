#include <cstdlib>
#include <exception>
#include <iostream>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif
#include <vulkan/vulkan_static_assertions.hpp>

import game;

int main() {  // NOLINT(bugprone-exception-escape): std::cerr is not configured to throw exceptions
  try {
    gfx::Game game;
    game.Start();
  } catch (const std::system_error& e) {
    std::cerr << '[' << e.code() << "] " << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "An unknown error occurred\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
