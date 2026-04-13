#include "scene/Scene.h"

uint32_t Scene::totalTriangles() const {
    uint32_t total = 0;
    for (auto& m : m_meshes)
        total += (uint32_t)m.indices.size() / 3;
    return total;
}

uint32_t Scene::totalVertices() const {
    uint32_t total = 0;
    for (auto& m : m_meshes)
        total += (uint32_t)m.positions.size();
    return total;
}
