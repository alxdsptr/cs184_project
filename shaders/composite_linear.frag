#version 450
// NRD+DLSS mode: combine denoised diffuse + specular + albedo + emissive into
// a linear HDR image at render resolution. DLSS will consume this next, then
// a separate tonemap pass maps to sRGB at output resolution.

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D uDiff;
layout(set = 0, binding = 1) uniform sampler2D uSpec;
layout(set = 0, binding = 2) uniform sampler2D uAlb;
layout(set = 0, binding = 3) uniform sampler2D uEmis;

void main() {
    vec3 diff = texture(uDiff, vUV).rgb;   // demodulated
    vec3 spec = texture(uSpec, vUV).rgb;
    vec3 alb  = texture(uAlb,  vUV).rgb;
    vec3 emis = texture(uEmis, vUV).rgb;

    vec3 hdr = diff * alb + spec + emis;

    // No tonemap, no gamma — output linear HDR directly.
    fragColor = vec4(hdr, 1.0);
}
