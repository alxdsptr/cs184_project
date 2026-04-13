#ifndef CGL_CUDA_SCENE_H
#define CGL_CUDA_SCENE_H

#include <vector>
#include <map>
#include "cuda_types.h"

// Forward declarations from CGL
namespace CGL {
    class Camera;
    class BSDF;
    namespace SceneObjects {
        struct BVHNode;
        class BVHAccel;
        class SceneLight;
        struct Scene;
    }
}

// Linearize pointer-based BVH tree into flat arrays for GPU.
// Returns the total number of BVH nodes written.
int cuda_flatten_bvh(
    const CGL::SceneObjects::BVHNode* root,
    std::vector<CUDABVHNode>& out_nodes,
    std::vector<CUDAPrimitive>& out_primitives,
    std::map<CGL::BSDF*, int>& material_map
);

// Build GPU material array from the BSDF pointer -> index map.
void cuda_build_materials(
    const std::map<CGL::BSDF*, int>& material_map,
    std::vector<CUDAMaterial>& out_materials
);

// Build GPU light array from scene lights.
void cuda_build_lights(
    const std::vector<CGL::SceneObjects::SceneLight*>& lights,
    std::vector<CUDALight>& out_lights
);

// Extract camera parameters into a flat GPU struct.
CUDACameraParams cuda_build_camera(
    const CGL::Camera* camera,
    int screen_w, int screen_h
);

#endif // CGL_CUDA_SCENE_H
