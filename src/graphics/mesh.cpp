#include "graphics/mesh.h"

#include <stdexcept>

namespace {

/**
 * \brief Ensures the provided vertex positions, texture coordinates, normals, and element indices describe a
 *        triangle mesh in addition to enforcing alignment between vertex attribute.
 */
void Validate(const std::span<const glm::vec3> positions,
              const std::span<const glm::vec2> texture_coordinates,
              const std::span<const glm::vec3> normals,
              const std::span<const GLuint> indices) {
  if (positions.empty()) {
    throw std::invalid_argument{"Vertex positions must be specified"};
  }
  if ((indices.empty() && positions.size() % 3 != 0) || indices.size() % 3 != 0) {
    throw std::invalid_argument{"Object must be a triangle mesh"};
  }
  if ((indices.empty() && !texture_coordinates.empty()) && positions.size() != texture_coordinates.size()) {
    throw std::invalid_argument{"Texture coordinates must align with position data"};
  }
  if (indices.empty() && !normals.empty() && positions.size() != normals.size()) {
    throw std::invalid_argument{"Vertex normals must align with position data"};
  }
}

}  // namespace

qem::Mesh::Mesh(const std::span<const glm::vec3> positions,
                const std::span<const glm::vec2> texture_coordinates,
                const std::span<const glm::vec3> normals,
                const std::span<const GLuint> indices,
                const glm::mat4& model_transform)
    : positions_{positions.begin(), positions.end()},
      texture_coordinates_{texture_coordinates.begin(), texture_coordinates.end()},
      normals_{normals.begin(), normals.end()},
      indices_{indices.begin(), indices.end()},
      model_transform_{model_transform} {
  Validate(positions_, texture_coordinates_, normals_, indices_);

  glGenVertexArrays(1, &vertex_array_);
  glBindVertexArray(vertex_array_);

  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);

  using PositionType = decltype(positions_)::value_type;
  using TextureCoordinateType = decltype(texture_coordinates_)::value_type;
  using NormalType = decltype(normals_)::value_type;

  // allocate memory for the vertex buffer
  const auto positions_size = static_cast<GLsizeiptr>(sizeof(PositionType) * positions_.size());
  const auto texture_coords_size = static_cast<GLsizeiptr>(sizeof(TextureCoordinateType) * texture_coordinates_.size());
  const auto normals_size = static_cast<GLsizeiptr>(sizeof(NormalType) * normals_.size());
  const auto buffer_size = positions_size + texture_coords_size + normals_size;
  glNamedBufferStorage(vertex_buffer_, buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT);

  // copy positions to the vertex buffer
  glNamedBufferSubData(vertex_buffer_, 0, positions_size, positions_.data());
  glVertexAttribPointer(0, PositionType::length(), GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  // copy texture coordinates to the vertex buffer
  if (!texture_coordinates_.empty()) {
    glNamedBufferSubData(vertex_buffer_, positions_size, texture_coords_size, texture_coordinates_.data());
    glVertexAttribPointer(1, TextureCoordinateType::length(), GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
  }

  // copy normals to the vertex buffer
  if (!normals_.empty()) {
    glNamedBufferSubData(vertex_buffer_, positions_size + texture_coords_size, normals_size, normals_.data());
    glVertexAttribPointer(2, NormalType::length(), GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(2);
  }

  // copy indices to the element buffer
  if (!indices_.empty()) {
    const auto indices_size = static_cast<GLsizeiptr>(sizeof(decltype(indices_)::value_type) * indices_.size());
    glGenBuffers(1, &element_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_);
    glNamedBufferStorage(element_buffer_, indices_size, indices_.data(), GL_MAP_READ_BIT);
  }
}

qem::Mesh& qem::Mesh::operator=(Mesh&& mesh) noexcept {
  if (this != &mesh) {
    std::swap(vertex_array_, mesh.vertex_array_);
    std::swap(vertex_buffer_, mesh.vertex_buffer_);
    std::swap(element_buffer_, mesh.element_buffer_);
    std::swap(positions_, mesh.positions_);
    std::swap(texture_coordinates_, mesh.texture_coordinates_);
    std::swap(normals_, mesh.normals_);
    std::swap(indices_, mesh.indices_);
    std::swap(model_transform_, mesh.model_transform_);
  }
  return *this;
}

qem::Mesh::~Mesh() noexcept {
  glDeleteVertexArrays(1, &vertex_array_);
  glDeleteBuffers(1, &vertex_buffer_);
  glDeleteBuffers(1, &element_buffer_);
}
