#ifndef SRC_GRAPHICS_OBJ_LOADER_H_
#define SRC_GRAPHICS_OBJ_LOADER_H_

#include <filesystem>

namespace qem {
class Mesh;

namespace obj_loader {

/**
 * \brief Loads a triangle mesh from an .obj file.
 * \param filepath The path to the .obj file.
 * \return A mesh defined by the position, texture coordinates, normals, and indices specified in the .obj file.
 * \note At this time, only a subset of .obj file specification is supported. Specifically, only 3D vertex
 *       position, 2D texture coordinates, and 3D normals are supported. Face elements are supported and may
 *       optionally contain texture coordinate and normal indices.
 * \throw std::invalid_argument Thrown if the file format is unsupported.
 * \throw std::runtime_error Thrown if the file cannot be opened.
 * \see https://en.wikipedia.org/wiki/Wavefront_.obj_file
 */
Mesh LoadMesh(const std::filesystem::path& filepath);

}  // namespace obj_loader
}  // namespace qem

#endif  // SRC_GRAPHICS_OBJ_LOADER_H_
