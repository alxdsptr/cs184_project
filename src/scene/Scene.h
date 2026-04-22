#pragma once
#include "accel/AABB.h"
#include "scene/AreaLight.h"
#include "scene/Mesh.h"
#include "scene/Material.h"
#include "scene/Light.h"
#include <vector>

// One emissive-mesh "light object" the debug UI can box + toggle. Populated
// at scene upload time — each entry groups the contiguous range of area-light
// triangles that came from the same source mesh, plus its world-space AABB.
// `enabled == false` suppresses both NEE sampling and direct BSDF hits for
// every triangle in this range.
struct EmissiveMeshInfo {
    AABB bounds;
    uint32_t areaLightStart = 0;   // first index into Scene::m_areaLights
    uint32_t areaLightCount = 0;
    int      materialIndex  = -1;  // dominant material (for debug label)
    bool     enabled        = true;
};

struct SceneCamera {
    bool  valid = false;
    float3 position = make_float3(0.0f, 1.0f, 3.0f);
    float3 forward = make_float3(0.0f, 0.0f, -1.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float  horizontalFovRadians = 60.0f * 3.14159265358979323846f / 180.0f;
    float  aspect = 0.0f;
    float  nearPlane = 0.1f;
    float  farPlane = 1000.0f;
};

class Scene {
public:
    std::vector<TriangleMesh>& getMeshes() { return m_meshes; }
    std::vector<PBRMaterial>&  getMaterials() { return m_materials; }
    std::vector<PointLight>&   getLights() { return m_lights; }
    std::vector<TriangleAreaLight>& getAreaLights() { return m_areaLights; }
    std::vector<EmissiveMeshInfo>& getEmissiveMeshes() { return m_emissiveMeshes; }
    const std::vector<TriangleMesh>& getMeshes() const { return m_meshes; }
    const std::vector<PBRMaterial>&  getMaterials() const { return m_materials; }
    const std::vector<PointLight>&   getLights() const { return m_lights; }
    const std::vector<TriangleAreaLight>& getAreaLights() const { return m_areaLights; }
    const std::vector<EmissiveMeshInfo>& getEmissiveMeshes() const { return m_emissiveMeshes; }

    AABB& getBounds() { return m_bounds; }
    const AABB& getBounds() const { return m_bounds; }

    SceneCamera& getCamera() { return m_camera; }
    const SceneCamera& getCamera() const { return m_camera; }

    uint32_t totalTriangles() const;
    uint32_t totalVertices() const;

private:
    std::vector<TriangleMesh> m_meshes;
    std::vector<PBRMaterial>  m_materials;
    std::vector<PointLight>   m_lights;
    std::vector<TriangleAreaLight> m_areaLights;
    std::vector<EmissiveMeshInfo>  m_emissiveMeshes;
    AABB m_bounds;
    SceneCamera m_camera;
};
