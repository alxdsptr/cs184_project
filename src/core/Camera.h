#pragma once
#include "core/Types.h"
#include "core/Math.h"
#include <string>

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
    bool up       = false;
    bool down     = false;
    float mouseDx = 0.0f;
    float mouseDy = 0.0f;
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
    // Call EXACTLY once per displayed frame, after the renderer has consumed
    // CameraParams. Snapshots the current view/proj as "prev" for the next
    // frame's motion vectors. `getParams()` itself is now side-effect-free,
    // so GUI/overlay code may call it freely without disturbing temporal
    // reprojection (the bug that broke ReSTIR PT/GI history when the normal-
    // arrow overlay was on).
    void advanceFrame();
    bool hasMoved() const { return m_moved; }
    void setAspect(float a);

    float3 getPosition() const { return m_position; }
    float  getFovDeg() const { return m_fovDeg; }
    float  getYawDeg() const { return m_yaw; }
    float  getPitchDeg() const { return m_pitch; }
    float  getAspect() const { return m_aspect; }
    float  getNearPlane() const { return m_nearPlane; }
    float  getFarPlane() const { return m_farPlane; }
    float  getMoveSpeed() const { return m_moveSpeed; }
    void   setMoveSpeed(float s) { m_moveSpeed = clampf(s, 0.05f, 200.0f); }

    void lockMovementFrame();
    void unlockMovementFrame() { m_frameLocked = false; }
    bool isMovementFrameLocked() const { return m_frameLocked; }

    // ── Deterministic auto-motion (for benchmarking / captures) ───────────
    // Two modes available:
    //
    //   AUTO_DOLLY  (default for capture mode) — slides the camera along its
    //     `m_forward` axis at constant speed. The forward direction is frozen
    //     at enable time so the camera keeps pointing in whatever direction
    //     the user/scene placed it. Best for stress-testing ReSTIR temporal
    //     reprojection: translation along the view axis is sympathetic to
    //     normal-dot and position-drift gates, so most pixels keep history.
    //
    //   AUTO_ORBIT  — circles `center` on a horizontal arc of `radius` with
    //     `pitchDeg` elevation. Always looks at the centre. Stresses the
    //     reprojection gates much harder; useful when you want to see how
    //     ReSTIR copes with continuous rotation.
    //
    // Both are pure functions of their elapsed time, so identical capture
    // settings → identical frame sequence between runs → meaningful per-mode
    // quality comparison.
    void setAutoDolly(float speedUnitsPerSec);
    void setAutoOrbit(float3 center, float radius, float periodSeconds,
                      float pitchDeg = 15.0f);
    void clearAutoMotion() { m_autoMotion = AutoMotion::None; }
    bool isAutoMoving() const { return m_autoMotion != AutoMotion::None; }
    // Resets phase so subsequent updates trace a reproducible path regardless
    // of when auto-motion was enabled in the session.
    void resetAutoMotionPhase() { m_autoMotionElapsed = 0.0f; }

    bool saveToFile(const std::string& path) const;
    bool loadFromFile(const std::string& path);
    // Symmetric in-memory write of all v1 camera fields. Marks the camera as
    // moved so the renderer drops temporal history. Used by the replay loop
    // (Application::runReplay) to inject one pose per frame without touching
    // the disk.
    void setPose(float3 position, float yawDeg, float pitchDeg,
                 float fovDeg, float aspect, float nearPlane, float farPlane);
    // Like setPose but does NOT overwrite the prev-frame matrices, so the
    // next renderFrame still produces correct motion vectors against whatever
    // prev was before the call. Caller is expected to have invoked
    // advanceFrame() first to snapshot the previous pose. Used by the replay
    // loop to thread motion vectors between successive recorded poses;
    // setPose() is the teleport variant that intentionally zeroes them.
    void setPosePreserveHistory(float3 position, float yawDeg, float pitchDeg,
                                float fovDeg, float aspect,
                                float nearPlane, float farPlane);

private:
    void rebuildMatrices();

    float3 m_position = make_float3(0, 0, 3);
    float  m_yaw   = -90.0f;  // degrees
    float  m_pitch = 0.0f;
    float  m_fovDeg   = 60.0f;
    float  m_aspect   = 16.0f / 9.0f;
    float  m_nearPlane = 0.01f;
    float  m_farPlane  = 1000.0f;
    // Default move speed — was 3.0; lowered to 0.9 (=30%) per user feedback;
    // 3.0 was uncomfortably fast for fine-grained inspection.
    float  m_moveSpeed  = 0.9f;
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

    // Auto-motion state. When m_autoMotion != None, update() ignores input.
    enum class AutoMotion { None, Dolly, Orbit };
    AutoMotion m_autoMotion = AutoMotion::None;
    float      m_autoMotionElapsed = 0.0f;

    // Dolly: linear slide along a frozen forward direction.
    float3 m_autoDollyOrigin    = make_float3(0, 0, 0);
    float3 m_autoDollyDirection = make_float3(0, 0, -1);
    float  m_autoDollySpeed     = 0.5f;

    // Orbit: rotate around centre with pitch elevation, looking at centre.
    float3 m_autoOrbitCenter    = make_float3(0, 0, 0);
    float  m_autoOrbitRadius    = 1.0f;
    float  m_autoOrbitPeriod    = 8.0f;
    float  m_autoOrbitPitchDeg  = 15.0f;
};
