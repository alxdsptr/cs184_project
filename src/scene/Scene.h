#pragma once
#include "scene/Mesh.h"
#include "scene/Material.h"
#include <vector>

class Scene {
public:
    std::vector<TriangleMesh>& getMeshes() { return m_meshes; }
    std::vector<PBRMaterial>&  getMaterials() { return m_materials; }
    const std::vector<TriangleMesh>& getMeshes() const { return m_meshes; }
    const std::vector<PBRMaterial>&  getMaterials() const { return m_materials; }

    uint32_t totalTriangles() const;
    uint32_t totalVertices() const;

private:
    std::vector<TriangleMesh> m_meshes;
    std::vector<PBRMaterial>  m_materials;
};
