#version 450
// Post-DLSS tonemap pass (NRD+DLSS mode): linear HDR → sRGB LDR at output res.
// Tonemap operator mirrors src/render/Tonemapping.cu for visual consistency.

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D uHDR;

layout(push_constant) uniform PC {
    float exposure;
    int   tonemapMode;  // 0 = None, 1 = Reinhard, 2 = ACES
} pc;

vec3 acesTonemap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 reinhardTonemap(vec3 x) {
    return x / (vec3(1.0) + x);
}

float linearToSRGB(float x) {
    return (x <= 0.0031308) ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

vec3 linearToSRGB(vec3 c) {
    return vec3(linearToSRGB(c.x), linearToSRGB(c.y), linearToSRGB(c.z));
}

void main() {
    vec3 hdr = texture(uHDR, vUV).rgb * pc.exposure;
    vec3 ldr = hdr;
    if (pc.tonemapMode == 1)      ldr = reinhardTonemap(hdr);
    else if (pc.tonemapMode == 2) ldr = acesTonemap(hdr);
    fragColor = vec4(linearToSRGB(ldr), 1.0);
}
