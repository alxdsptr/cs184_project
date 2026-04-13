#pragma once
#include "scene/AreaLight.h"
#include "scene/Mesh.h"
#include "scene/Material.h"
#include "scene/Light.h"
#include <vector>

class Scene {
public:
    std::vector<TriangleMesh>& getMeshes() { return m_meshes; }
    std::vector<PBRMaterial>&  getMaterials() { return m_materials; }
    std::vector<PointLight>&   getLights() { return m_lights; }
    std::vector<TriangleAreaLight>& getAreaLights() { return m_areaLights; }
    const std::vector<TriangleMesh>& getMeshes() const { return m_meshes; }
    const std::vector<PBRMaterial>&  getMaterials() const { return m_materials; }
    const std::vector<PointLight>&   getLights() const { return m_lights; }
    const std::vector<TriangleAreaLight>& getAreaLights() const { return m_areaLights; }

    uint32_t totalTriangles() const;
    uint32_t totalVertices() const;

private:
    std::vector<TriangleMesh> m_meshes;
    std::vector<PBRMaterial>  m_materials;
    std::vector<PointLight>   m_lights;
    std::vector<TriangleAreaLight> m_areaLights;
};
