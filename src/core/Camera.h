#pragma once
#include "core/Types.h"
#include "core/Math.h"

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
    bool hasMoved() const { return m_moved; }
    void setAspect(float a);

    float3 getPosition() const { return m_position; }
    float  getFovDeg() const { return m_fovDeg; }
    float  getMoveSpeed() const { return m_moveSpeed; }
    void   setMoveSpeed(float s) { m_moveSpeed = clampf(s, 0.05f, 200.0f); }

    void lockMovementFrame();
    void unlockMovementFrame() { m_frameLocked = false; }
    bool isMovementFrameLocked() const { return m_frameLocked; }

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
};
