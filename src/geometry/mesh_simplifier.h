#ifndef SRC_GEOMETRY_MESH_SIMPLIFIER_H_
#define SRC_GEOMETRY_MESH_SIMPLIFIER_H_

namespace qem {
class Mesh;

namespace mesh {

/**
 * \brief Reduces the number of triangles in a mesh.
 * \param mesh The mesh to simplify.
 * \param rate The percentage of triangles to be removed (e.g., .95 indicates 95% of triangles should be removed).
 * \return A triangle mesh with \p rate percent of triangles removed from \p mesh.
 * \throw std::invalid_argument Thrown if the simplification rate is not in the interval [0,1].
 * \see docs/surface_simplification for a detailed description of this mesh simplification algorithm.
 */
Mesh Simplify(const Mesh& mesh, float rate);

}  // namespace mesh
}  // namespace qem

#endif  // SRC_GEOMETRY_MESH_SIMPLIFIER_H_
