#include "core/Camera.h"
#include "core/Halton.h"
#include "core/SceneCollider.h"
#include "util/Log.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

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
    m_lifetimeT += dt;

    // Mouse look — same in both modes; gated by mouseHeld so the caller can
    // disable look while the cursor is released for ImGui interaction.
    if (input.mouseHeld && (fabsf(input.mouseDx) > 0.01f || fabsf(input.mouseDy) > 0.01f)) {
        m_yaw   += input.mouseDx * m_mouseSens;
        m_pitch -= input.mouseDy * m_mouseSens;
        m_pitch  = clampf(m_pitch, -89.0f, 89.0f);
        m_moved  = true;
        rebuildMatrices(); // refresh basis so movement uses current heading
    }

    if (m_collider && m_collider->ready()) {
        updateCollider(dt, input);
    } else {
        updateFreeFly(dt, input);
    }

    if (m_moved) {
        rebuildMatrices();
    }
}

void Camera::updateFreeFly(float dt, const InputState& input) {
    float speed = m_moveSpeed * dt;

    // WASD movement — use locked basis if the user has frozen the movement
    // frame, otherwise follow the current camera orientation.
    const float3 fwd    = m_frameLocked ? m_lockedForward : m_forward;
    const float3 right  = m_frameLocked ? m_lockedRight   : m_right;
    const float3 upAxis = m_frameLocked ? m_lockedUp      : make_float3(0, 1, 0);

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
}

bool Camera::probeGround(float3 from, float maxDist, float& groundY) const {
    if (!m_collider || !m_collider->ready()) return false;
    float t;
    float3 n;
    // Start the probe just above `from` so we don't miss a face we're already
    // resting on (eps offset survives floating-point jitter).
    float3 origin = make_float3(from.x, from.y + 0.05f, from.z);
    if (m_collider->raycast(origin, make_float3(0, -1, 0), maxDist + 0.05f, t, n)) {
        groundY = origin.y - t;
        return true;
    }
    return false;
}

float3 Camera::sweepHorizontal(float3 from, float3 delta) const {
    // `delta` should be horizontal; we sweep along X then Z so the camera
    // slides along walls instead of binding on inner corners. For each axis
    // we cast a ray of length |delta| + radius and clamp to the hit.
    float3 result = make_float3(0, 0, 0);
    float3 cur = from;

    auto sweepAxis = [&](int axis) {
        float d = (axis == 0) ? delta.x : delta.z;
        if (fabsf(d) < 1e-6f) return;
        float3 dir = make_float3(0, 0, 0);
        if (axis == 0) dir.x = (d > 0) ? 1.0f : -1.0f;
        else           dir.z = (d > 0) ? 1.0f : -1.0f;

        float reach = fabsf(d) + m_collisionRadius;
        float tHit;
        float3 n;
        if (m_collider->raycast(cur, dir, reach, tHit, n)) {
            // Stop `m_collisionRadius` before the hit so the capsule's edge
            // sits on the wall. Negative results clamp to zero (already
            // touching).
            float allowed = fmaxf(0.0f, tHit - m_collisionRadius);
            float step = (d > 0) ? fminf(allowed, d) : fmaxf(-allowed, d);
            if (axis == 0) { cur.x += step; result.x = step; }
            else           { cur.z += step; result.z = step; }
        } else {
            if (axis == 0) { cur.x += d; result.x = d; }
            else           { cur.z += d; result.z = d; }
        }
    };

    sweepAxis(0);
    sweepAxis(2);
    return result;
}

void Camera::updateCollider(float dt, const InputState& input) {
    // Cap dt so a hitch doesn't tunnel us through the floor / a wall.
    if (dt > 0.05f) dt = 0.05f;

    // ── Double-tap jump → toggle fly mode ──
    if (input.jumpPressed) {
        float since = m_lifetimeT - m_lastJumpPressTime;
        if (since < 0.30f) {
            m_flyMode   = !m_flyMode;
            m_velocityY = 0.0f;
            LOG_INFO("Camera: fly mode %s", m_flyMode ? "ON" : "OFF");
            // Consume both presses so a triple-tap doesn't toggle twice.
            m_lastJumpPressTime = -1e9f;
        } else {
            m_lastJumpPressTime = m_lifetimeT;
        }
    }

    // ── Build horizontal move from WASD ──
    // Project camera basis onto the XZ plane so pitch doesn't drag us into
    // the floor or sky when walking.
    float3 fwdH = make_float3(m_forward.x, 0.0f, m_forward.z);
    float3 rgtH = make_float3(m_right.x,   0.0f, m_right.z);
    float fl = length(fwdH);
    float rl = length(rgtH);
    if (fl > 1e-6f) fwdH = fwdH / fl;
    else            fwdH = make_float3(0, 0, -1);
    if (rl > 1e-6f) rgtH = rgtH / rl;
    else            rgtH = make_float3(1, 0, 0);

    float3 moveDir = make_float3(0, 0, 0);
    if (input.forward)  moveDir += fwdH;
    if (input.backward) moveDir += fwdH * (-1.0f);
    if (input.left)     moveDir += rgtH * (-1.0f);
    if (input.right)    moveDir += rgtH;
    float ml = length(moveDir);
    if (ml > 1e-6f) moveDir = moveDir / ml;

    float speedScale = m_flyMode ? 2.0f : 1.0f;
    float3 horizDelta = moveDir * (m_moveSpeed * speedScale * dt);

    if (length(horizDelta) > 1e-6f) {
        float3 applied = sweepHorizontal(m_position, horizDelta);
        if (length(applied) > 1e-6f) {
            m_position += applied;
            m_moved = true;
        }
    }

    // ── Vertical motion ──
    if (m_flyMode) {
        // Creative-mode: free vertical translation, no gravity, no ground
        // snap. Hold space to ascend, shift to descend.
        float v = 0.0f;
        if (input.up)   v += 1.0f;
        if (input.down) v -= 1.0f;
        if (fabsf(v) > 1e-6f) {
            m_position.y += v * m_moveSpeed * speedScale * dt;
            m_moved = true;
        }
        m_velocityY = 0.0f;
        m_onGround = false;
    } else {
        // Walking: gravity, jump on rising-edge of space, ground snap.
        float groundY = 0.0f;
        bool onGround = probeGround(m_position - make_float3(0, m_eyeHeight, 0),
                                    0.20f, groundY);
        // probeGround took feet-level origin; convert ground hit back to eye Y.
        float groundEyeY = groundY + m_eyeHeight;

        if (onGround && m_velocityY <= 0.0f) {
            m_position.y = groundEyeY;
            m_velocityY = 0.0f;
            m_onGround = true;
        } else {
            m_onGround = false;
        }

        if (input.jumpPressed && m_onGround) {
            m_velocityY = m_jumpSpeed;
            m_onGround = false;
        }

        // Integrate gravity.
        m_velocityY -= m_gravity * dt;
        // Clamp terminal velocity to keep tunneling under control on cheap
        // raycasts.
        if (m_velocityY < -50.0f) m_velocityY = -50.0f;

        float dy = m_velocityY * dt;
        if (fabsf(dy) > 1e-6f) {
            // Cast against ceiling/floor before applying so we don't poke
            // through low ceilings or fall through thin floors.
            float3 dirV = make_float3(0, dy > 0 ? 1.0f : -1.0f, 0);
            float reach = fabsf(dy) + m_collisionRadius;
            float tHit;
            float3 n;
            float3 origin = m_position;
            if (m_collider->raycast(origin, dirV, reach, tHit, n)) {
                float allowed = fmaxf(0.0f, tHit - m_collisionRadius);
                float step = (dy > 0) ? fminf(allowed, dy) : fmaxf(-allowed, dy);
                m_position.y += step;
                if (fabsf(step) < fabsf(dy) - 1e-5f) {
                    // Hit something — kill vertical velocity.
                    m_velocityY = 0.0f;
                    if (dy < 0.0f) m_onGround = true;
                }
            } else {
                m_position.y += dy;
            }
            m_moved = true;
        }
    }
}

void Camera::snapToGround(float maxDrop) {
    if (!m_collider || !m_collider->ready()) return;
    float groundY;
    // Probe from well above current position to find the floor underneath.
    float3 origin = make_float3(m_position.x,
                                m_position.y + 0.5f,
                                m_position.z);
    if (probeGround(origin, maxDrop, groundY)) {
        m_position = make_float3(m_position.x,
                                 groundY + m_eyeHeight,
                                 m_position.z);
        m_velocityY = 0.0f;
        m_onGround = true;
        m_groundedOnce = true;
        m_flyMode = false;
        rebuildMatrices();
        m_moved = true;
    } else {
        LOG_WARN("Camera::snapToGround: no ground found beneath (%g, %g, %g)",
                 m_position.x, m_position.y, m_position.z);
    }
}

void Camera::lockMovementFrame() {
    m_lockedForward = m_forward;
    m_lockedRight   = m_right;
    m_lockedUp      = m_up;
    m_frameLocked   = true;
}

bool Camera::saveToFile(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERROR("Camera::saveToFile: could not open '%s' for writing", path.c_str());
        return false;
    }
    out << "# path_tracer camera v1\n";
    out << "position " << m_position.x << " " << m_position.y << " " << m_position.z << "\n";
    out << "yaw " << m_yaw << "\n";
    out << "pitch " << m_pitch << "\n";
    out << "fov_deg " << m_fovDeg << "\n";
    out << "aspect " << m_aspect << "\n";
    out << "near " << m_nearPlane << "\n";
    out << "far " << m_farPlane << "\n";
    return out.good();
}

bool Camera::loadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        LOG_ERROR("Camera::loadFromFile: could not open '%s'", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "position") {
            iss >> m_position.x >> m_position.y >> m_position.z;
        } else if (key == "yaw") {
            iss >> m_yaw;
        } else if (key == "pitch") {
            iss >> m_pitch;
        } else if (key == "fov_deg") {
            iss >> m_fovDeg;
        } else if (key == "aspect") {
            iss >> m_aspect;
        } else if (key == "near") {
            iss >> m_nearPlane;
        } else if (key == "far") {
            iss >> m_farPlane;
        }
    }
    rebuildMatrices();
    m_prevViewProj = mat4_multiply(m_projMatrix, m_viewMatrix);
    m_prevViewMatrix = m_viewMatrix;
    m_prevProjMatrix = m_projMatrix;
    m_moved = true;
    return true;
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
