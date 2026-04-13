#include "core/Camera.h"
#include "core/Halton.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void Camera::init(float3 pos, float3 target, float fovDeg, float aspect) {
    m_position = pos;
    m_fovDeg = fovDeg;
    m_aspect = aspect;

    float3 dir = normalize(target - pos);
    m_yaw   = atan2f(dir.z, dir.x) * 180.0f / (float)M_PI;
    m_pitch = asinf(clampf(dir.y, -0.999f, 0.999f)) * 180.0f / (float)M_PI;

    rebuildMatrices();
    m_prevViewProj = mat4_multiply(m_projMatrix, m_viewMatrix);
}

void Camera::update(float dt, const InputState& input) {
    m_moved = false;
    float speed = m_moveSpeed * dt;

    // WASD movement
    float3 moveDir = make_float3(0, 0, 0);
    if (input.forward)  moveDir += m_forward;
    if (input.backward) moveDir += m_forward * (-1.0f);
    if (input.left)     moveDir += m_right * (-1.0f);
    if (input.right)    moveDir += m_right;
    if (input.up)       moveDir += make_float3(0, 1, 0);
    if (input.down)     moveDir += make_float3(0, -1, 0);

    float len = length(moveDir);
    if (len > 0.001f) {
        m_position += moveDir * (speed / len);
        m_moved = true;
    }

    // Mouse look
    if (input.mouseHeld && (fabsf(input.mouseDx) > 0.01f || fabsf(input.mouseDy) > 0.01f)) {
        m_yaw   += input.mouseDx * m_mouseSens;
        m_pitch -= input.mouseDy * m_mouseSens;
        m_pitch  = clampf(m_pitch, -89.0f, 89.0f);
        m_moved  = true;
    }

    if (m_moved) {
        rebuildMatrices();
    }
}

void Camera::rebuildMatrices() {
    float yawRad   = m_yaw   * (float)M_PI / 180.0f;
    float pitchRad = m_pitch * (float)M_PI / 180.0f;

    m_forward = normalize(make_float3(
        cosf(yawRad) * cosf(pitchRad),
        sinf(pitchRad),
        sinf(yawRad) * cosf(pitchRad)
    ));
    m_right = normalize(cross(m_forward, make_float3(0, 1, 0)));
    m_up    = cross(m_right, m_forward);

    m_viewMatrix = mat4_lookAt(m_position, m_position + m_forward, make_float3(0, 1, 0));
    m_projMatrix = mat4_perspective(m_fovDeg * (float)M_PI / 180.0f, m_aspect, m_nearPlane, m_farPlane);
}

CameraParams Camera::getParams(uint32_t frameIndex) const {
    CameraParams p{};
    p.position     = m_position;
    p.forward      = m_forward;
    p.right        = m_right;
    p.up           = m_up;
    p.fovYRadians  = m_fovDeg * (float)M_PI / 180.0f;
    p.aspectRatio  = m_aspect;
    p.nearPlane    = m_nearPlane;
    p.farPlane     = m_farPlane;
    p.viewMatrix   = m_viewMatrix;
    p.projMatrix   = m_projMatrix;
    p.frameIndex   = frameIndex;

    // Jittered view-proj for rendering
    float2 jitter = haltonJitter(frameIndex);
    p.jitterOffset = jitter;

    // Apply sub-pixel jitter to projection matrix
    float4x4 jitteredProj = m_projMatrix;
    // Jitter offset is in pixel units; convert to NDC for the projection matrix
    // Not applied here (done in kernel via ray offset) -- store for DLSS
    p.viewProjMatrix    = mat4_multiply(m_projMatrix, m_viewMatrix);
    p.prevViewProjMatrix = m_prevViewProj;

    // Update prev for next frame (const_cast needed since we cache prev)
    const_cast<Camera*>(this)->m_prevViewProj = p.viewProjMatrix;

    return p;
}
