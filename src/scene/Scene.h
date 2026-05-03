#pragma once
#include "accel/AABB.h"
#include "core/VolumeMedium.h"
#include "scene/AreaLight.h"
#include "scene/Mesh.h"
#include "scene/Material.h"
#include "scene/Light.h"
#include <vector>

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
    std::vector<DirectionalLight>& getDirectionalLights() { return m_directionalLights; }
    std::vector<TriangleAreaLight>& getAreaLights() { return m_areaLights; }
    const std::vector<TriangleMesh>& getMeshes() const { return m_meshes; }
    const std::vector<PBRMaterial>&  getMaterials() const { return m_materials; }
    const std::vector<PointLight>&   getLights() const { return m_lights; }
    const std::vector<DirectionalLight>& getDirectionalLights() const { return m_directionalLights; }
    const std::vector<TriangleAreaLight>& getAreaLights() const { return m_areaLights; }

    AABB& getBounds() { return m_bounds; }
    const AABB& getBounds() const { return m_bounds; }

    SceneCamera& getCamera() { return m_camera; }
    const SceneCamera& getCamera() const { return m_camera; }

    HomogeneousMedium& getMedium() { return m_medium; }
    const HomogeneousMedium& getMedium() const { return m_medium; }

    uint32_t totalTriangles() const;
    uint32_t totalVertices() const;

private:
    std::vector<TriangleMesh> m_meshes;
    std::vector<PBRMaterial>  m_materials;
    std::vector<PointLight>   m_lights;
    std::vector<DirectionalLight> m_directionalLights;
    std::vector<TriangleAreaLight> m_areaLights;
    AABB m_bounds;
    SceneCamera m_camera;
    HomogeneousMedium m_medium;
};
