#version 450
// NRD-only mode: combine denoised diffuse + specular + emissive + albedo,
// tonemap, and output sRGB LDR.
//
// Inputs come from the NRD denoiser + the path-trace kernel:
//   uDiff  : denoised diffuse radiance (demodulated by albedo)
//   uSpec  : denoised specular radiance (already contains Fresnel/albedo)
//   uAlb   : diffuse albedo (for remodulation)
//   uEmis  : emissive HDR (not denoised)
// Tonemap operator must stay in lockstep with src/render/Tonemapping.cu so
// that NRDOnly parity holds against Native on a converged input.

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D uDiff;
layout(set = 0, binding = 1) uniform sampler2D uSpec;
layout(set = 0, binding = 2) uniform sampler2D uAlb;
layout(set = 0, binding = 3) uniform sampler2D uEmis;

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
    vec3 diff = texture(uDiff, vUV).rgb;   // demodulated
    vec3 spec = texture(uSpec, vUV).rgb;
    vec4 alb4 = texture(uAlb,  vUV);       // RGBA8 unorm; .a = surface-valid mask
    vec3 alb  = alb4.rgb;
    vec3 emis = texture(uEmis, vUV).rgb;

    // Remodulate diffuse by albedo; specular already lives in radiance space.
    // Mask both NRD outputs by alb.a (kernel writes 0 for sky pixels, 1 for
    // primary-hit pixels). NRD's RELAX leaves OUT_SPEC populated with stale
    // prev-frame values at sky pixels (some internal stage doesn't honor
    // viewZ > denoisingRange), which would otherwise produce visible ghost
    // trails on the env when the camera moves. Emissive is added unmasked
    // because the kernel writes envColor freshly for sky pixels every frame.
    vec3 hdr = (diff * alb + spec) * alb4.a + emis;

    // Exposure.
    hdr *= pc.exposure;

    // Tonemap.
    vec3 ldr = hdr;
    if (pc.tonemapMode == 1)      ldr = reinhardTonemap(hdr);
    else if (pc.tonemapMode == 2) ldr = acesTonemap(hdr);

    fragColor = vec4(linearToSRGB(ldr), 1.0);
}
