#pragma once
#include "core/Types.h"
#include "core/Math.h"
#include <string>

class SceneCollider;

struct CameraParams {
    float3   position;
    float    fovYRadians;
    float3   forward;
    float    aspectRatio;
    float3   right;
    float    nearPlane;
    float3   up;
    float    farPlane;
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float4x4 viewProjMatrix;
    float4x4 prevViewProjMatrix;
    // Previous-frame view & proj (unjittered), for denoisers (NRD) that want
    // them split. `prevViewProjMatrix` above remains valid for anyone who
    // only needs the combined form.
    float4x4 prevViewMatrix;
    float4x4 prevProjMatrix;
    float2   jitterOffset;
    uint32_t frameIndex;
    uint32_t _pad;
};

struct InputState {
    bool forward  = false;
    bool backward = false;
    bool left     = false;
    bool right    = false;
    // `up`/`down` keep the legacy free-fly meaning (translate along world up).
    // In collider mode `up` is the held jump key (ground jump or fly-mode
    // ascend), `down` is the descend key (fly mode only / sneak).
    bool up       = false;
    bool down     = false;
    // Rising edge of the jump key. Used for jump impulse + double-tap fly
    // toggle in collider mode.
    bool jumpPressed = false;
    float mouseDx = 0.0f;
    float mouseDy = 0.0f;
    // Apply `mouseDx/Dy` to look around. In free-fly mode this is the legacy
    // "right-mouse held" gate; in collider/cursor-captured mode it's true
    // every frame (cursor is hidden + locked).
    bool mouseHeld = false;
};

class Camera {
public:
    void init(float3 pos, float3 target, float fovDeg, float aspect);
    void init(float3 pos, float3 forward, float3 up, float fovDeg, float aspect);
    void update(float dt, const InputState& input);
    void setFovDeg(float fovDeg);
    void setClipPlanes(float nearPlane, float farPlane);

    CameraParams getParams(uint32_t frameIndex) const;
    bool hasMoved() const { return m_moved; }
    void setAspect(float a);

    float3 getPosition() const { return m_position; }
    float  getFovDeg() const { return m_fovDeg; }
    float  getMoveSpeed() const { return m_moveSpeed; }
    void   setMoveSpeed(float s) { m_moveSpeed = clampf(s, 0.05f, 200.0f); }

    void lockMovementFrame();
    void unlockMovementFrame() { m_frameLocked = false; }
    bool isMovementFrameLocked() const { return m_frameLocked; }

    // Attach a collider for FPS-style movement (gravity, jump, wall sweeps).
    // When null, `update()` falls back to the original free-fly behavior.
    void setCollider(const SceneCollider* c) { m_collider = c; }
    bool hasCollider() const { return m_collider != nullptr; }

    // Snap the camera to the ground directly below `m_position`. Resets fly
    // mode and vertical velocity. No-op if no collider is attached or if no
    // ground is found within `maxDrop`.
    void snapToGround(float maxDrop = 1000.0f);

    bool isFlying() const { return m_flyMode; }
    void setFlying(bool fly) { m_flyMode = fly; m_velocityY = 0.0f; }

    bool saveToFile(const std::string& path) const;
    bool loadFromFile(const std::string& path);

private:
    void rebuildMatrices();

    float3 m_position = make_float3(0, 0, 3);
    float  m_yaw   = -90.0f;  // degrees
    float  m_pitch = 0.0f;
    float  m_fovDeg   = 60.0f;
    float  m_aspect   = 16.0f / 9.0f;
    float  m_nearPlane = 0.01f;
    float  m_farPlane  = 1000.0f;
    float  m_moveSpeed  = 3.0f;
    float  m_mouseSens  = 0.15f;

    float3   m_forward = make_float3(0, 0, -1);
    float3   m_right   = make_float3(1, 0, 0);
    float3   m_up      = make_float3(0, 1, 0);
    float4x4 m_viewMatrix  = float4x4::identity();
    float4x4 m_projMatrix  = float4x4::identity();
    float4x4 m_prevViewProj = float4x4::identity();
    float4x4 m_prevViewMatrix = float4x4::identity();
    float4x4 m_prevProjMatrix = float4x4::identity();
    bool     m_moved = false;

    // Locked movement frame: when enabled, WASD/space/shift translate along
    // these saved axes instead of the current camera basis. Mouse-look still
    // rotates the view freely.
    bool   m_frameLocked = false;
    float3 m_lockedForward = make_float3(0, 0, -1);
    float3 m_lockedRight   = make_float3(1, 0, 0);
    float3 m_lockedUp      = make_float3(0, 1, 0);

    // ── Collider-mode physics (Minecraft-creative-style movement) ──
    // Active only when m_collider != nullptr. The camera is treated as a
    // capsule of half-width `m_collisionRadius` with eyes `m_eyeHeight`
    // above its feet; horizontal motion sweeps against scene geometry,
    // vertical motion is gravity-driven on the ground and free in fly mode.
    void updateFreeFly(float dt, const InputState& input);
    void updateCollider(float dt, const InputState& input);
    // Sweep a horizontal step `delta` (Y component is ignored), returning a
    // collision-clamped delta. Performs per-axis raycasts so we slide along
    // walls instead of stopping dead.
    float3 sweepHorizontal(float3 from, float3 delta) const;
    // Returns true if a downward ray from `from` hits ground within `maxDist`.
    // On hit, writes the ground Y-coordinate into `groundY`.
    bool   probeGround(float3 from, float maxDist, float& groundY) const;

    const SceneCollider* m_collider = nullptr;
    float  m_collisionRadius = 0.3f;   // half-width of the camera "capsule"
    float  m_eyeHeight       = 1.7f;   // distance from feet to eyes
    float  m_gravity         = 28.0f;  // m/s^2 — snappier than real gravity
    float  m_jumpSpeed       = 8.5f;   // m/s, gives ~1.3 m jump height
    float  m_velocityY       = 0.0f;
    bool   m_onGround        = false;
    bool   m_flyMode         = false;
    float  m_lastJumpPressTime = -1e9f;
    float  m_lifetimeT       = 0.0f;   // seconds since camera created
    bool   m_groundedOnce    = false;  // becomes true after first ground snap
};
