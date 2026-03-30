#include "bvh.h"

#include "CGL/CGL.h"
#include "triangle.h"

#include <algorithm>
#include <iostream>
#include <stack>

using namespace std;

namespace CGL {
namespace SceneObjects {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  primitives = std::vector<Primitive *>(_primitives);
  root = construct_bvh(primitives.begin(), primitives.end(), max_leaf_size);
}

BVHAccel::~BVHAccel() {
  if (root)
    delete root;
  primitives.clear();
}

BBox BVHAccel::get_bbox() const { return root->bb; }

void BVHAccel::draw(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->draw(c, alpha);
    }
  } else {
    draw(node->l, c, alpha);
    draw(node->r, c, alpha);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->drawOutline(c, alpha);
    }
  } else {
    drawOutline(node->l, c, alpha);
    drawOutline(node->r, c, alpha);
  }
}

BVHNode *BVHAccel::construct_bvh(std::vector<Primitive *>::iterator start,
                                 std::vector<Primitive *>::iterator end,
                                 size_t max_leaf_size) {

  // TODO (Part 2.1):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.
  size_t n_prims = end - start;

  BBox bbox;
  BBox centroid_bbox;
  for (auto p = start; p != end; p++) {
    const BBox &pb = (*p)->get_bbox();
    bbox.expand(pb);
    centroid_bbox.expand(pb.centroid());
  }

  BVHNode *node = new BVHNode(bbox);

  size_t leaf_size = max((size_t)1, max_leaf_size);
  if (n_prims <= leaf_size) {
    node->start = start;
    node->end = end;
    return node;
  }

  Vector3D extent = centroid_bbox.extent;
  int axis = 0;
  if (extent.y > extent.x && extent.y >= extent.z) {
    axis = 1;
  } else if (extent.z > extent.x && extent.z > extent.y) {
    axis = 2;
  }

  double split = 0.0;
  for (auto p = start; p != end; p++) {
    split += (*p)->get_bbox().centroid()[axis];
  }
  split /= n_prims;

  auto mid = std::partition(start, end, [axis, split](Primitive *p) {
    return p->get_bbox().centroid()[axis] < split;
  });

  if (mid == start || mid == end) {
    mid = start + n_prims / 2;
    std::nth_element(start, mid, end, [axis](Primitive *a, Primitive *b) {
      return a->get_bbox().centroid()[axis] < b->get_bbox().centroid()[axis];
    });
  }

  node->l = construct_bvh(start, mid, max_leaf_size);
  node->r = construct_bvh(mid, end, max_leaf_size);
  return node;
}

bool BVHAccel::has_intersection(const Ray &ray, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
  // Take note that this function has a short-circuit that the
  // Intersection version cannot, since it returns as soon as it finds
  // a hit, it doesn't actually have to find the closest hit.

  if (node == NULL) {
    return false;
  }

  double t0 = ray.min_t;
  double t1 = ray.max_t;
  if (!node->bb.intersect(ray, t0, t1)) {
    return false;
  }

  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      total_isects++;
      if ((*p)->has_intersection(ray)) {
        return true;
      }
    }
    return false;
  }

  if (has_intersection(ray, node->l)) {
    return true;
  }

  return has_intersection(ray, node->r);


}

bool BVHAccel::intersect(const Ray &ray, Intersection *i, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.

  if (node == NULL) {
    return false;
  }

  double t0 = ray.min_t;
  double t1 = ray.max_t;
  if (!node->bb.intersect(ray, t0, t1)) {
    return false;
  }

  if (node->isLeaf()) {
    bool hit = false;
    for (auto p = node->start; p != node->end; p++) {
      total_isects++;
      hit = (*p)->intersect(ray, i) || hit;
    }
    return hit;
  }

  bool hit_l = intersect(ray, i, node->l);
  bool hit_r = intersect(ray, i, node->r);
  return hit_l || hit_r;
}

} // namespace SceneObjects
} // namespace CGL
