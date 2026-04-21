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

void Camera::init(float3 pos, float3 forward, float3 up, float fovDeg, float aspect) {
    m_position = pos;
    m_fovDeg = fovDeg;
    m_aspect = aspect;

    m_forward = normalize(forward);
    m_right = normalize(cross(m_forward, up));
    if (length(m_right) <= 1e-6f) {
        m_right = make_float3(1, 0, 0);
    }
    m_up = normalize(cross(m_right, m_forward));
    if (length(m_up) <= 1e-6f) {
        m_up = make_float3(0, 1, 0);
    }

    m_yaw   = atan2f(m_forward.z, m_forward.x) * 180.0f / (float)M_PI;
    m_pitch = asinf(clampf(m_forward.y, -0.999f, 0.999f)) * 180.0f / (float)M_PI;

    m_viewMatrix = mat4_lookAt(m_position, m_position + m_forward, m_up);
    m_projMatrix = mat4_perspective(m_fovDeg * (float)M_PI / 180.0f, m_aspect, m_nearPlane, m_farPlane);
    m_prevViewProj = mat4_multiply(m_projMatrix, m_viewMatrix);
}

void Camera::setFovDeg(float fovDeg) {
    m_fovDeg = clampf(fovDeg, 5.0f, 120.0f);
    rebuildMatrices();
    m_moved = true;
}

void Camera::setClipPlanes(float nearPlane, float farPlane) {
    m_nearPlane = fmaxf(1e-4f, nearPlane);
    m_farPlane = fmaxf(m_nearPlane + 1e-4f, farPlane);
    rebuildMatrices();
    m_moved = true;
}

void Camera::setAspect(float a) {
    m_aspect = a;
    rebuildMatrices();
    m_moved = true;
}

void Camera::update(float dt, const InputState& input) {
    m_moved = false;
    float speed = m_moveSpeed * dt;

    // WASD movement — use locked basis if the user has frozen the movement
    // frame, otherwise follow the current camera orientation.
    const float3 fwd   = m_frameLocked ? m_lockedForward : m_forward;
    const float3 right = m_frameLocked ? m_lockedRight   : m_right;
    const float3 upAxis = m_frameLocked ? m_lockedUp     : make_float3(0, 1, 0);

    float3 moveDir = make_float3(0, 0, 0);
    if (input.forward)  moveDir += fwd;
    if (input.backward) moveDir += fwd * (-1.0f);
    if (input.left)     moveDir += right * (-1.0f);
    if (input.right)    moveDir += right;
    if (input.up)       moveDir += upAxis;
    if (input.down)     moveDir += upAxis * (-1.0f);

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

void Camera::lockMovementFrame() {
    m_lockedForward = m_forward;
    m_lockedRight   = m_right;
    m_lockedUp      = m_up;
    m_frameLocked   = true;
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
    p.prevViewMatrix     = m_prevViewMatrix;
    p.prevProjMatrix     = m_prevProjMatrix;

    // Update prev for next frame (const_cast needed since we cache prev)
    Camera* self = const_cast<Camera*>(this);
    self->m_prevViewProj    = p.viewProjMatrix;
    self->m_prevViewMatrix  = m_viewMatrix;
    self->m_prevProjMatrix  = m_projMatrix;

    return p;
}
