#include "cuda_scene.h"
#include "cuda_types.h"

#include "scene/bvh.h"
#include "scene/triangle.h"
#include "scene/sphere.h"
#include "scene/light.h"
#include "scene/scene.h"
#include "pathtracer/bsdf.h"
#include "pathtracer/camera.h"

#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"

#include <cmath>

using namespace CGL;
using namespace CGL::SceneObjects;

// ============================================================
// Helpers
// ============================================================

static inline float3 to_f3(const Vector3D& v) {
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}

// ============================================================
// BVH linearization
// ============================================================

static int flatten_bvh_recursive(
    const BVHNode* node,
    std::vector<CUDABVHNode>& out_nodes,
    std::vector<CUDAPrimitive>& out_primitives,
    std::map<BSDF*, int>& material_map)
{
    int my_index = (int)out_nodes.size();
    out_nodes.push_back({});

    CUDABVHNode& gpu_node = out_nodes[my_index];
    gpu_node.bb_min = to_f3(node->bb.min);
    gpu_node.bb_max = to_f3(node->bb.max);
    gpu_node.pad = 0;

    if (node->isLeaf()) {
        gpu_node.is_leaf = 1;
        gpu_node.left_or_first = (int)out_primitives.size();
        int count = 0;

        for (auto it = node->start; it != node->end; ++it) {
            CUDAPrimitive gp;
            memset(&gp, 0, sizeof(gp));

            BSDF* bsdf = (*it)->get_bsdf();
            // Register material if not already
            if (bsdf && material_map.find(bsdf) == material_map.end()) {
                material_map[bsdf] = (int)material_map.size();
            }
            int mat_id = bsdf ? material_map[bsdf] : 0;

            if (auto* tri = dynamic_cast<Triangle*>(*it)) {
                gp.type = PRIM_TRIANGLE;
                gp.tri.p1 = to_f3(tri->p1);
                gp.tri.p2 = to_f3(tri->p2);
                gp.tri.p3 = to_f3(tri->p3);
                gp.tri.n1 = to_f3(tri->n1);
                gp.tri.n2 = to_f3(tri->n2);
                gp.tri.n3 = to_f3(tri->n3);
                gp.tri.material_id = mat_id;
            } else if (auto* sph = dynamic_cast<Sphere*>(*it)) {
                gp.type = PRIM_SPHERE;
                gp.sph.center = to_f3(sph->o);
                gp.sph.radius = (float)sph->r;
                gp.sph.radius2 = (float)sph->r2;
                gp.sph.material_id = mat_id;
            }

            out_primitives.push_back(gp);
            count++;
        }
        gpu_node.right_or_count = count;
    } else {
        gpu_node.is_leaf = 0;

        int left_idx = flatten_bvh_recursive(node->l, out_nodes, out_primitives, material_map);
        // Re-grab reference since vector may have reallocated
        out_nodes[my_index].left_or_first = left_idx;

        int right_idx = flatten_bvh_recursive(node->r, out_nodes, out_primitives, material_map);
        out_nodes[my_index].right_or_count = right_idx;
    }

    return my_index;
}

int cuda_flatten_bvh(
    const BVHNode* root,
    std::vector<CUDABVHNode>& out_nodes,
    std::vector<CUDAPrimitive>& out_primitives,
    std::map<BSDF*, int>& material_map)
{
    out_nodes.clear();
    out_primitives.clear();
    flatten_bvh_recursive(root, out_nodes, out_primitives, material_map);
    return (int)out_nodes.size();
}

// ============================================================
// Material extraction
// ============================================================

void cuda_build_materials(
    const std::map<BSDF*, int>& material_map,
    std::vector<CUDAMaterial>& out_materials)
{
    out_materials.resize(material_map.size());

    for (auto& [bsdf, idx] : material_map) {
        CUDAMaterial& mat = out_materials[idx];
        memset(&mat, 0, sizeof(mat));

        if (auto* d = dynamic_cast<DiffuseBSDF*>(bsdf)) {
            mat.type = MAT_DIFFUSE;
            mat.is_delta = 0;
            // Access reflectance: DiffuseBSDF stores it as private member.
            // We use f() with wo=(0,0,1), wi=(0,0,1) which returns reflectance/PI
            Vector3D r = d->f(Vector3D(0, 0, 1), Vector3D(0, 0, 1));
            // reflectance = r * PI
            mat.albedo = to_f3(r * PI);
        } else if (auto* e = dynamic_cast<EmissionBSDF*>(bsdf)) {
            mat.type = MAT_EMISSION;
            mat.is_delta = 0;
            mat.emission = to_f3(e->get_emission());
        } else if (auto* m = dynamic_cast<MirrorBSDF*>(bsdf)) {
            mat.type = MAT_MIRROR;
            mat.is_delta = 1;
            // MirrorBSDF has reflectance but it's private and f() returns zero.
            // For now, use a default white reflectance (stubs in original code)
            mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
        } else if (auto* g = dynamic_cast<GlassBSDF*>(bsdf)) {
            mat.type = MAT_GLASS;
            mat.is_delta = 1;
            mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
        } else if (auto* r = dynamic_cast<RefractionBSDF*>(bsdf)) {
            mat.type = MAT_REFRACTION;
            mat.is_delta = 1;
        } else if (auto* mf = dynamic_cast<MicrofacetBSDF*>(bsdf)) {
            mat.type = MAT_MICROFACET;
            mat.is_delta = 0;
        } else {
            // Unknown BSDF, default to diffuse gray
            mat.type = MAT_DIFFUSE;
            mat.is_delta = 0;
            mat.albedo = make_float3(0.5f, 0.5f, 0.5f);
        }
    }
}

// ============================================================
// Light extraction
// ============================================================

void cuda_build_lights(
    const std::vector<SceneLight*>& lights,
    std::vector<CUDALight>& out_lights)
{
    out_lights.clear();

    for (auto* light : lights) {
        CUDALight gl;
        memset(&gl, 0, sizeof(gl));

        if (auto* dl = dynamic_cast<DirectionalLight*>(light)) {
            gl.type = LIGHT_DIRECTIONAL;
            gl.is_delta = 1;
            // Extract radiance and direction via sample_L since members are private
            Vector3D wi; double dist, pdf;
            Vector3D rad = dl->sample_L(Vector3D(0,0,0), &wi, &dist, &pdf);
            gl.radiance = to_f3(rad);
            gl.dir_to_light = to_f3(wi);
            out_lights.push_back(gl);
        } else if (auto* pl = dynamic_cast<PointLight*>(light)) {
            gl.type = LIGHT_POINT;
            gl.is_delta = 1;
            gl.radiance = to_f3(pl->radiance);
            gl.position = to_f3(pl->position);
            out_lights.push_back(gl);
        } else if (auto* al = dynamic_cast<AreaLight*>(light)) {
            gl.type = LIGHT_AREA;
            gl.is_delta = 0;
            gl.radiance = to_f3(al->radiance);
            gl.position = to_f3(al->position);
            gl.area_direction = to_f3(al->direction);
            gl.area_dim_x = to_f3(al->dim_x);
            gl.area_dim_y = to_f3(al->dim_y);
            gl.area = (float)al->area;
            out_lights.push_back(gl);
        } else if (auto* sl = dynamic_cast<SpotLight*>(light)) {
            gl.type = LIGHT_SPOT;
            gl.is_delta = 1;
            gl.radiance = to_f3(sl->radiance);
            gl.position = to_f3(sl->position);
            gl.spot_direction = to_f3(sl->direction);
            gl.spot_angle = (float)sl->angle;
            out_lights.push_back(gl);
        } else if (auto* ihl = dynamic_cast<InfiniteHemisphereLight*>(light)) {
            gl.type = LIGHT_INF_HEMISPHERE;
            gl.is_delta = 0;
            gl.radiance = to_f3(ihl->radiance);
            out_lights.push_back(gl);
        }
        // Skip SphereLight, MeshLight, EnvironmentLight for GPU v1
    }
}

// ============================================================
// Camera extraction
// ============================================================

CUDACameraParams cuda_build_camera(
    const Camera* camera,
    int screen_w, int screen_h)
{
    CUDACameraParams cam;
    memset(&cam, 0, sizeof(cam));

    cam.pos = to_f3(camera->position());

    const Matrix3x3& c2w = camera->camera_to_world();
    cam.c2w_col0 = to_f3(c2w[0]);
    cam.c2w_col1 = to_f3(c2w[1]);
    cam.c2w_col2 = to_f3(c2w[2]);

    double hFov = camera->h_fov();
    double vFov = camera->v_fov();
    cam.hFov_half_tan = (float)tan(hFov * PI / 360.0);  // tan(degrees_to_rad / 2)
    cam.vFov_half_tan = (float)tan(vFov * PI / 360.0);

    cam.nClip = (float)camera->near_clip();
    cam.fClip = (float)camera->far_clip();
    cam.lensRadius = (float)camera->lensRadius;
    cam.focalDistance = (float)camera->focalDistance;
    cam.screenW = screen_w;
    cam.screenH = screen_h;

    return cam;
}
