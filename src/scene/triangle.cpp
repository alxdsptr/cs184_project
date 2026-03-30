#include "triangle.h"

#include "CGL/CGL.h"
#include "GL/glew.h"

namespace CGL {
namespace SceneObjects {

Triangle::Triangle(const Mesh *mesh, size_t v1, size_t v2, size_t v3) {
  p1 = mesh->positions[v1];
  p2 = mesh->positions[v2];
  p3 = mesh->positions[v3];
  n1 = mesh->normals[v1];
  n2 = mesh->normals[v2];
  n3 = mesh->normals[v3];
  bbox = BBox(p1);
  bbox.expand(p2);
  bbox.expand(p3);

  bsdf = mesh->get_bsdf();
}

BBox Triangle::get_bbox() const { return bbox; }

bool Triangle::has_intersection(const Ray &r) const {
  // Part 1, Task 3: implement ray-triangle intersection
  // The difference between this function and the next function is that the next
  // function records the "intersection" while this function only tests whether
  // there is a intersection.

  // Möller-Trumbore algorithm
  Vector3D e1 = p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s1 = cross(r.d, e2);
  double a = dot(e1, s1);

  if (std::abs(a) < 1e-8) {
    return false;
  }

  Vector3D s = r.o - p1;
  double f = 1.0 / a;
  double u = f * dot(s, s1);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  Vector3D s2 = cross(s, e1);
  double v = f * dot(r.d, s2);

  // Check if v is in valid range [0, 1] and u + v <= 1
  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  double t = f * dot(e2, s2);

  // Check if t is in valid range [min_t, max_t]
  return t >= r.min_t && t <= r.max_t;
}

bool Triangle::intersect(const Ray &r, Intersection *isect) const {
  // Part 1, Task 3:
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly

  // Möller-Trumbore algorithm
  Vector3D e1 = p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s1 = cross(r.d, e2);
  double a = dot(e1, s1);

  // If a is near zero, ray is parallel to triangle
  if (std::abs(a) < 1e-8) {
    return false;
  }

  Vector3D s = r.o - p1;
  double f = 1.0 / a;
  double u = f * dot(s, s1);

  // Check if u is in valid range [0, 1]
  if (u < 0.0 || u > 1.0) {
    return false;
  }

  Vector3D s2 = cross(s, e1);
  double v = f * dot(r.d, s2);

  // Check if v is in valid range [0, 1] and u + v <= 1
  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  double t = f * dot(e2, s2);

  // Check if t is in valid range [min_t, max_t]
  if (t < r.min_t || t > r.max_t) {
    return false;
  }

  // Update intersection information
  isect->t = t;
  isect->primitive = this;

  // Interpolate normals using barycentric coordinates
  // The barycentric coordinates are (w, u, v) where w = 1 - u - v
  double w = 1.0 - u - v;
  isect->n = w * n1 + u * n2 + v * n3;

  // Get the BSDF
  isect->bsdf = get_bsdf();

  // Update the ray's max_t to be the nearest intersection
  r.max_t = t;

  return true;

}

void Triangle::draw(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_TRIANGLES);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

void Triangle::drawOutline(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_LINE_LOOP);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

} // namespace SceneObjects
} // namespace CGL
