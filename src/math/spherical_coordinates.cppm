module;

#include <cmath>

#include <glm/glm.hpp>

export module spherical_coordiantes;

namespace gfx {

export struct SphericalCoordinates {
  float radius = 0.0f;
  float theta = 0.0f;
  float phi = 0.0f;
};

export [[nodiscard]] SphericalCoordinates ToSphericalCoordinates(const glm::vec3& cartesian_coordinates);
export [[nodiscard]] glm::vec3 ToCartesianCoordinates(const SphericalCoordinates& spherical_coordinates);

}  // namespace gfx

module :private;

namespace gfx {

SphericalCoordinates ToSphericalCoordinates(const glm::vec3& cartesian_coordinates) {
  const auto radius = glm::length(cartesian_coordinates);
  return radius == 0.0f ? SphericalCoordinates{.radius = 0.0f, .theta = 0.0f, .phi = 0.0f}
                        : SphericalCoordinates{.radius = radius,
                                               .theta = std::atan2(cartesian_coordinates.x, cartesian_coordinates.z),
                                               .phi = std::asin(-cartesian_coordinates.y / radius)};
}

glm::vec3 ToCartesianCoordinates(const SphericalCoordinates& spherical_coordinates) {
  const auto& [radius, theta, phi] = spherical_coordinates;
  const auto cos_phi = std::cos(phi);
  const auto x = radius * std::sin(theta) * cos_phi;
  const auto y = radius * std::sin(-phi);
  const auto z = radius * std::cos(theta) * cos_phi;
  return glm::vec3{x, y, z};
}

}  // namespace gfx
