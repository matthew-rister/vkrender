#include "geometry/vertex.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "geometry/half_edge.h"

namespace {

TEST(VertexTest, InitializationSetsTheVertexId) {
  constexpr auto kId = 7;
  const qem::Vertex vertex{kId, glm::vec3{0.0}};
  EXPECT_EQ(vertex.id(), kId);
}

TEST(VertexTest, InitializationSetsTheVertexPosition) {
  constexpr glm::vec3 kPosition{1.0f, 2.0f, 3.0f};
  const qem::Vertex vertex{kPosition};
  EXPECT_EQ(vertex.position(), kPosition);
}

TEST(VertexTest, SetIdUpdatesTheVertexId) {
  constexpr auto kId = 7;
  qem::Vertex vertex{0, glm::vec3{0.0f}};
  vertex.set_id(kId);
  EXPECT_EQ(vertex.id(), kId);
}

TEST(VertexTest, SetEdgeUpdatesTheVertexHalfEdge) {
  const auto vertex = std::make_shared<qem::Vertex>(7, glm::vec3{0.0f});
  const auto edge = std::make_shared<qem::HalfEdge>(vertex);
  vertex->set_edge(edge);
  EXPECT_EQ(vertex->edge(), edge);
}

TEST(VertexTest, EqualVerticesHaveTheSameHashValue) {
  const qem::Vertex vertex{0, glm::vec3{0.0f}};
  const auto vertex_copy = vertex;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(vertex, vertex_copy);
  EXPECT_EQ(hash_value(vertex), hash_value(vertex_copy));
}

TEST(VertexTest, EqualVertexPairsHaveTheSameHashValue) {
  const qem::Vertex v0{0, glm::vec3{0.0f}};
  const qem::Vertex v1{1, glm::vec3{0.0f}};
  EXPECT_EQ(hash_value(v0, v1), hash_value(qem::Vertex{v0}, qem::Vertex{v1}));
}

TEST(VertexTest, EqualVertexTriplesHaveTheSameHashValue) {
  const qem::Vertex v0{0, glm::vec3{0.0f}};
  const qem::Vertex v1{1, glm::vec3{0.0f}};
  const qem::Vertex v2{2, glm::vec3{0.0f}};
  EXPECT_EQ(hash_value(v0, v1, v2), hash_value(qem::Vertex{v0}, qem::Vertex{v1}, qem::Vertex{v2}));
}

TEST(VertexTest, FlipVertexPairsDoNotHaveTheSameHashValue) {
  const qem::Vertex v0{0, glm::vec3{0.0f}};
  const qem::Vertex v1{1, glm::vec3{0.0f}};
  EXPECT_NE(hash_value(v0, v1), hash_value(v1, v0));
}

#ifndef NDEBUG

TEST(VertexTest, GetUnsetIdCausesProgramExit) {
  const qem::Vertex vertex{glm::vec3{}};
  EXPECT_DEATH({ std::ignore = vertex.id(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

TEST(VertexTest, GetExpiredEdgeCausesProgramExit) {
  const auto vertex = std::make_shared<qem::Vertex>(0, glm::vec3{0.0f});
  {
    const auto edge = std::make_shared<qem::HalfEdge>(vertex);
    vertex->set_edge(edge);
  }
  EXPECT_DEATH({ std::ignore = vertex->edge(); }, "");  // NOLINT(whitespace/newline) NOLINT(whitespace/comments)
}

#endif

}  // namespace
