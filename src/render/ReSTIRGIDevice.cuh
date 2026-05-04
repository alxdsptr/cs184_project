#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// GRIS device-side helpers (Lin et al. 2022, "Generalized Resampled Importance
// Sampling: Foundations of ReSTIR"). Header-only so every TU sees identical
// definitions.
//
// Despite the historical filename ("GI"), this header now hosts the full GRIS
// reservoir machinery used by both ReSTIR GI (k=1 path postfix) and ReSTIR PT
// (k>=1 path postfix). The two algorithms differ only in initial-candidate
// generation; reservoir merging, MIS weighting, the reconnection-shift
// Jacobian, and the integral estimator are shared.
//
// ── Notation ────────────────────────────────────────────────────────────────
//   X_i               input candidate sample (paper §4.2)
//   T_i               shift mapping (paper §4.3) — here, the reconnection
//                     shift: pixel q' reuses sample (x_r, n_r, L_o) by
//                     re-shading f_r(q', q'->x_r) at the destination visible
//                     point (paper §7.4)
//   p̂                target function = luminance(f_r * L_o) * cos(θ_q)
//   p̂_{←i}(y)        the source i's target function pulled to the destination
//                     domain via T_i^{-1}: p̂_i(T_i^{-1}(y)) · |∂T_i^{-1}/∂y|.
//                     For the reconnection shift this evaluates p̂ at peer i's
//                     visible point with the SAME (x_r, n_r, L_o), times the
//                     reverse Jacobian.
//   W_i               unbiased contribution weight (paper Eq. 22)
//   M_i               confidence weight (paper §6.2). NOT a sample count —
//                     it weights the sample's contribution to MIS.
//   wSum              Σ w_i across all merged sources for the current
//                     reservoir; the reservoir's `W` reconstructs as
//                     wSum / p̂(Y) (paper Eq. 22). M does NOT appear in this
//                     reconstruction; it only appears inside m_i.
//
// ── Resampling MIS (paper §5.6, defensive pairwise variant Eq. 38) ─────────
//
// We use defensive pairwise MIS with |R| = 1 (the destination pixel is the
// single canonical source). For Mc = M_dst (held) and M = Σ M_i (across held
// and all peers), defining Mnon = M - Mc:
//
//   For the canonical (held) sample at its own location y_R = y_dst:
//
//     m_dst(y_R) = 1/M
//                + (1/M) · Σ_{j∈peers}  p̂(y_R)
//                                       ────────────────────────────────
//                                       Mc·p̂(y_R) + Mnon·p̂_{←j}(y_R)
//
//   For a peer i at its own (shifted) location y_i:
//
//     m_peer(y_i)  = (Mnon/M) · p̂_{←i}(y_i)
//                              ──────────────────────────────────
//                              Mc·p̂(y_i) + Mnon·p̂_{←i}(y_i)
//
// (Eq. 38 in the paper.) The sample's contribution to wSum is then
//     w_i = m_i(y_i) · p̂(y_i) · W_i · |∂T_i/∂x|
// with |∂T/∂x| = 1 for the held canonical sample. After all peers are
// processed:
//     dst.W = wSum / p̂(Y)        (Eq. 22 — NO factor of M)
//
// The paper proves (Theorem A.4) this scheme bounds the resampling weight by
// w_i ≤ C_i / |R|, which keeps Var[Σw_i] finite and lets us guarantee
// asymptotic convergence to the target distribution (Theorems A.2, A.3).
//
// ── Avoiding singularities (paper §5.4 and §7.4) ───────────────────────────
//   • Reconnection-shift Jacobian goes to zero (sample REJECTED, not
//     clamped) when geometric configuration violates bijectivity or makes
//     the ratio explode. Asymmetric clamps inject permanent bias (Fig. 10b).
//   • Connectability conditions (paper §7.5): minimum reconnection distance
//     to suppress 1/r² fireflies near corners. Roughness gating happens at
//     candidate-generation time (the caller filters by lobe selection).
// ─────────────────────────────────────────────────────────────────────────────

#include "render/ReSTIRGI.h"
#include "render/ReSTIRDevice.cuh"
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ── Cached source target accessor ───────────────────────────────────────────
// `gris_cHat(r)` returns p̂_i evaluated at sample y in the SOURCE pixel's
// measure (after the appropriate Jacobian for the shift map used to import
// the sample). For canonical samples freshly streamed at this pixel, c_i
// equals r.pHat. We piggyback on the unused _pad0 slot so the GIReservoir
// struct stays binary-compatible with prior cudaMalloc / OptiX launch
// params.
__device__ inline float& gris_cHat(GIReservoir& r)             { return r._pad0; }
__device__ inline const float& gris_cHat(const GIReservoir& r) { return r._pad0; }

// ── Reservoir lifecycle ─────────────────────────────────────────────────────
__device__ inline void giReservoirReset(GIReservoir& r) {
    r.visiblePos     = make_float3(0, 0, 0);
    r.visibleNormal  = make_float3(0, 1, 0);
    r.samplePos      = make_float3(0, 0, 0);
    r.sampleNormal   = make_float3(0, 1, 0);
    r.sampleRadiance = make_float3(0, 0, 0);
    r.pHat           = 0.0f;
    r.W              = 0.0f;
    r.M              = 0.0f;
    r.isEnv          = 0u;
    r.valid          = 0u;
    r._pad0          = 0.0f;     // cHat (cached source target)
    r.xrRoughness    = 1.0f;     // unknown → treat as fully rough
    r._pad2          = 0.0f;
}

// ── Reconnection-shift connectability (paper §7.5) ──────────────────────────
// Reconnections shorter than this distance introduce 1/r² singularities in
// area-form integrands; the paper recommends 1-5% of scene diameter. We use
// 2 cm as a robust default tuned to the bundled glTF assets (0.5-50 m).
#ifndef GRIS_RECONN_MIN_DIST
#define GRIS_RECONN_MIN_DIST 0.02f
#endif
// Per-paper §7.5 BSDF-roughness threshold below which the lobe is treated as
// near-specular and not safe to reconnect through. 0.2 GGX α matches the
// paper's recommendation; both the visible-point side AND the reconnection
// vertex must clear this threshold for reconnection to be valid. Below the
// threshold the shift is "undefined" (sample excluded from the destination's
// MIS sum) — random-replay would have to be used as a fallback to recover
// such samples, which we leave as future work (this matches Hua et al.'s
// hybrid shift; we currently implement the reconnection-only variant).
#ifndef GRIS_RECONN_MIN_ROUGHNESS
#define GRIS_RECONN_MIN_ROUGHNESS 0.2f
#endif

// Geometric outputs at a destination visible point q for a stored sample.
//   wi   : unit direction q → x_s (or sample direction for env)
//   r2   : squared distance q → x_s (1.0 for env)
//   cosQ : max(0, dot(n_q, wi))
//   cosS : max(0, dot(n_s, -wi))  (1.0 for env)
__device__ inline bool giConnect(
    const float3& q, const float3& nq,
    const GIReservoir& r,
    float3& wi, float& r2, float& cosQ, float& cosS)
{
    if (r.isEnv) {
        wi   = r.samplePos;
        r2   = 1.0f;
        cosQ = fmaxf(dot(nq, wi), 0.0f);
        cosS = 1.0f;
        return cosQ > 0.0f;
    }
    float3 d  = r.samplePos - q;
    float  d2 = dot(d, d);
    if (d2 < 1e-8f) return false;
    float invLen = rsqrtf(d2);
    wi  = d * invLen;
    r2  = d2;
    cosQ = fmaxf(dot(nq, wi), 0.0f);
    cosS = fmaxf(dot(r.sampleNormal, -wi), 0.0f);
    return (cosQ > 0.0f) && (cosS > 0.0f);
}

// Reconnection-shift connectability: paper §7.5 distance + roughness +
// orientation checks. When false, the shift is "undefined" for this
// (q, sample) pair and the sample must be excluded from the destination's
// MIS sum. Caller passes dst's surface roughness so we can also gate on
// the q' side (the reconnection vertex roughness lives in the reservoir).
__device__ inline bool giShiftConnectable(
    const float3& qDst, const float3& nqDst, float qDstRoughness,
    const GIReservoir& r)
{
    if (r.isEnv) return true;                  // env shift is identity
    // Roughness gate (paper §7.5): both visible-point side AND reconnection
    // vertex must be sufficiently rough to keep the path-contribution ratio
    // bounded. xrRoughness=0 treats the vertex as pure-diffuse (always OK).
    if (qDstRoughness < GRIS_RECONN_MIN_ROUGHNESS) return false;
    if (r.xrRoughness > 0.0f && r.xrRoughness < GRIS_RECONN_MIN_ROUGHNESS)
        return false;
    float3 d  = r.samplePos - qDst;
    float  d2 = dot(d, d);
    if (d2 < GRIS_RECONN_MIN_DIST * GRIS_RECONN_MIN_DIST) return false;
    float invLen = rsqrtf(d2);
    float3 wi = d * invLen;
    if (dot(nqDst, wi) <= 0.0f) return false;
    if (dot(r.sampleNormal, -wi) <= 0.0f) return false;
    return true;
}

// Target function p̂ at a destination visible point.
__device__ inline float giEvalTargetPdf(
    const ReSTIRSurface& surf,
    const GIReservoir&   r,
    float3& wi)
{
    float r2 = 0.0f, cosQ = 0.0f, cosS = 0.0f;
    if (!giConnect(surf.position, surf.normal, r, wi, r2, cosQ, cosS))
        return 0.0f;

    float3 brdf = restirEvalBrdf(surf, wi);
    float fLum  = restirLuminance(brdf);
    if (fLum <= 0.0f) return 0.0f;

    float Lum = restirLuminance(r.sampleRadiance);
    if (Lum <= 0.0f) return 0.0f;

    return fLum * Lum * cosQ;
}

// Reconnection-shift Jacobian (paper Eq. 52) for reusing a sample produced
// at q_src on dst's surface q_dst. Returns 0 when the configuration violates
// bijectivity or would inflate the ratio outside a sane range — these
// samples MUST be rejected, NOT clamped (paper §5.4).
__device__ inline float giJacobian(
    const float3& qDst,
    const float3& qSrc,
    const GIReservoir& r)
{
    if (r.isEnv) return 1.0f;
    float3 dDst = r.samplePos - qDst;
    float3 dSrc = r.samplePos - qSrc;
    float  r2Dst = dot(dDst, dDst);
    float  r2Src = dot(dSrc, dSrc);
    if (r2Dst < 1e-8f || r2Src < 1e-8f) return 0.0f;
    float invDst = rsqrtf(r2Dst);
    float invSrc = rsqrtf(r2Src);
    float cosSDst = fmaxf(dot(r.sampleNormal, -dDst * invDst), 0.0f);
    float cosSSrc = fmaxf(dot(r.sampleNormal, -dSrc * invSrc), 0.0f);
    if (cosSSrc <= 0.0f || cosSDst <= 0.0f) return 0.0f;
    float jac = (cosSDst * r2Src) / (cosSSrc * r2Dst);
    // REJECT — do NOT clamp — Jacobians outside [1/50, 50]. Asymmetric
    // clamps inject permanent bias (paper §5.4 explicitly warns against
    // it; Figure 10b shows the failure mode). The 50× window is wide
    // enough that almost no real-physics reconnection trips it; the few
    // that do are recovered by the canonical sample (paper §5.5).
    if (jac > 50.0f || jac < (1.0f / 50.0f)) return 0.0f;
    return jac;
}

// ─────────────────────────────────────────────────────────────────────────────
// GRIS reservoir streaming primitives
// ─────────────────────────────────────────────────────────────────────────────
//
// We expose two distinct primitives:
//
//   gris_streamCandidate :  add one independent canonical candidate to dst
//                           during initial-candidate RIS. The candidate is
//                           born at dst's surface, so its shift is identity
//                           and its MIS weight reduces to 1/M.
//
//   gris_mergeMultiPair  :  fold N peer reservoirs (from temporal/spatial
//                           neighbors) into dst, jointly MIS-weighted with
//                           defensive pairwise MIS (paper Eq. 38). Single-
//                           pass and order-independent in expectation.
//
// ── Defensive resampling-weight clamps (M7 flash-and-decay fix) ─────────────
// Same rationale as the DI clamps in ReSTIRDevice.cuh: with 9759+ small
// emissive triangles, BSDF-sampled candidates that land near-grazing on a
// triangle produce p̂/p_src ratios of order 1e6+ in fp32. Once such a sample
// wins RIS, GRIS's m_i · p̂(y_i) · W_i · jac propagates it through every
// peer's MIS denominator and the spatial/temporal merge keeps it dominant
// for ~mCap frames. Lin et al. §5.4 explicitly warns the resampling weight
// must be bounded for the convergence proof of Thm A.4 to hold; clamping
// here enforces that bound.
#ifndef RESTIR_GI_MAX_WCAND
#define RESTIR_GI_MAX_WCAND 1.0e4f
#endif
#ifndef RESTIR_GI_MAX_W
#define RESTIR_GI_MAX_W     1.0e4f
#endif
#ifndef RESTIR_GI_MAX_MERGE_W
#define RESTIR_GI_MAX_MERGE_W 1.0e6f
#endif

// Initial-candidate RIS finalisation. For an M-candidate canonical RIS pass
// (Talbot weights m_i = 1/M, paper Eq. 5):
//     W = (1/M) · Σ p̂/p_src / p̂(Y)
// `gris_streamCandidate` accumulates the unscaled Σ p̂/p_src; we apply the
// 1/M factor here. Used after EVERY initial-candidate stream.
__device__ inline void giReservoirFinalize(GIReservoir& r, float wSum) {
    if (!r.valid || r.pHat <= 0.0f || r.M <= 0.0f) {
        r.W = 0.0f;
        return;
    }
    float W = wSum / (r.pHat * r.M);
    // Bound the contribution weight: once W is large, the next frame's
    // temporal pass cap-then-merges it with new candidates, and the
    // multiplier `M·p̂·W` in pairwise MIS keeps the bad sample winning.
    // The spatial pass then broadcasts it to neighbors. Capping here is
    // the cleanest single point of defense that survives both the
    // temporal-cap displacement window AND the spatial broadcast.
    if (W > RESTIR_GI_MAX_W) W = RESTIR_GI_MAX_W;
    if (!isfinite(W)) W = 0.0f;
    r.W = W;
}

// Append a canonical candidate produced at the destination surface.
// In RIS over canonical samples, the resampling MIS weight collapses to
// 1/M (Talbot Eq. 5 with all p_i equal); we accumulate Σ p̂/p_src here and
// divide by M in the finalize step. To make the finalize formula uniform
// across both initial RIS and GRIS merges we instead bake the 1/M into the
// stream — at completion of the M-candidate loop the caller scales wSum
// by 1/M. We expose the unscaled accumulator so the caller can choose its
// own normalisation. Returns true if the candidate replaced the held sample.
__device__ inline bool gris_streamCandidate(
    GIReservoir& r, float& wSum,
    const float3& visiblePos, const float3& visibleNormal,
    bool isEnv, const float3& samplePos, const float3& sampleNormal,
    const float3& sampleRadiance,
    float pHat,        // p̂ at dst surface
    float pSrc,        // source PDF at dst (typically pdfBsdf)
    float u01)
{
    // Confidence increases by 1 unconditionally (paper §6.2 / §5.5: M is
    // the canonical-sample count for the convergence proof of §5.7).
    r.M += 1.0f;
    if (!(pSrc > 0.0f) || !(pHat > 0.0f)) return false;

    float w = pHat / pSrc;
    // Defensive clamp at the source: a near-zero pSrc (e.g. very narrow
    // GGX lobe BSDF sample landing on a near-grazing emitter) can blow w
    // up. Lin et al. Thm A.4 needs bounded w for Var[Σwᵢ] convergence;
    // we enforce that bound directly.
    if (w > RESTIR_GI_MAX_WCAND) w = RESTIR_GI_MAX_WCAND;
    if (!isfinite(w)) return false;
    wSum += w;
    if (u01 * wSum < w) {
        r.visiblePos     = visiblePos;
        r.visibleNormal  = visibleNormal;
        r.samplePos      = samplePos;
        r.sampleNormal   = sampleNormal;
        r.sampleRadiance = sampleRadiance;
        r.pHat           = pHat;
        gris_cHat(r)     = pHat;        // canonical: c_i ≡ p̂
        r.isEnv          = isEnv ? 1u : 0u;
        r.valid          = 1u;
        return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-peer GRIS merge with defensive pairwise MIS (paper Eq. 38).
//
// Inputs:
//   dst       : reservoir holding the destination's *canonical* sample
//               (its initial-candidate result for this frame). MAY be empty
//               (then the merge runs with |R|=0 effectively, falling back to
//               peer-only Talbot).
//   dstSurf   : the destination visible-point surface.
//   peers     : array of N peer reservoirs (from spatial neighbors and/or
//               the temporal history pixel).
//   peerSurf  : array of N peer surfaces, in the same order. REQUIRED for
//               the inner sum of the held term's MIS weight (we evaluate
//               p̂_{←j}(y_R) which means re-shading the held sample at peer
//               j's surface).
//   u01s      : array of N+1 uniform randoms — u01s[i] for peer i's RIS
//               step, u01s[N] currently unused (reserved for canonical
//               re-resampling if we add it).
//   numPeers  : N.
//
// Effect:
//   Replaces dst's held sample, pHat, W per the GRIS-resampled Y, and sets
//   dst.M = M_dst + Σ M_peer (post-cap by caller).
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline void gris_mergeMultiPair(
    GIReservoir& dst,
    const ReSTIRSurface& dstSurf,
    const GIReservoir* peers,
    const ReSTIRSurface* peerSurf,
    uint32_t numPeers,
    const float* u01s)
{
    // Snapshot the held sample (we'll re-evaluate its MIS weight against
    // the full peer set below).
    GIReservoir held = dst;
    bool heldValid = (held.valid && held.pHat > 0.0f && held.W > 0.0f);

    float Mc = held.M;                          // canonical confidence (|R|=1)
    float M  = Mc;
    for (uint32_t i = 0; i < numPeers; i++) M += peers[i].M;
    float Mnon = M - Mc;
    if (Mnon < 0.0f) Mnon = 0.0f;
    float invM = (M > 0.0f) ? (1.0f / M) : 0.0f;

    float wSum = 0.0f;

    // ── Held term: m_dst(y_R) per Eq. 38 (defensive variant) ────────────
    if (heldValid) {
        float pHatR = held.pHat;                // p̂(y_R) at dst's surface
        float m_dst = invM;                     // 1/M term
        if (Mnon > 0.0f) {
            // Inner sum: Σ_{j∈peers} p̂(y_R) / (Mc·p̂(y_R) + Mnon·p̂_{←j}(y_R))
            // p̂_{←j}(y_R) = p̂(y_R | evaluated at peer j's surface) ·
            //               jacobian(peer j ← held source)
            // Because the held sample's source IS dst, we shift to peer j
            // by treating peer j as the destination of the reconnection.
            float innerSum = 0.0f;
            for (uint32_t j = 0; j < numPeers; j++) {
                if (peerSurf[j].valid < 0.5f) continue;
                if (!giShiftConnectable(peerSurf[j].position,
                                         peerSurf[j].normal,
                                         peerSurf[j].roughness,
                                         held)) continue;

                float3 wiTmp;
                GIReservoir tmp = held;
                float pHatAtJ = giEvalTargetPdf(peerSurf[j], tmp, wiTmp);
                if (!(pHatAtJ > 0.0f)) continue;

                // Reverse Jacobian: source = dst (held.visiblePos),
                //                    dest   = peer j (peerSurf[j].position).
                float jacInv = giJacobian(peerSurf[j].position,
                                           held.visiblePos, held);
                if (!(jacInv > 0.0f)) continue;

                float c_at_R_via_j = pHatAtJ * jacInv;
                float denom = Mc * pHatR + Mnon * c_at_R_via_j;
                if (denom > 0.0f) {
                    innerSum += pHatR / denom;
                }
            }
            m_dst += invM * innerSum;
        }
        // Eq. 19 with identity shift: w_dst = m_dst · p̂(y_R) · W_dst.
        float w_dst = m_dst * pHatR * held.W;
        if (!isfinite(w_dst) || !(w_dst > 0.0f)) {
            w_dst = 0.0f;
        } else if (w_dst > RESTIR_GI_MAX_MERGE_W) {
            w_dst = RESTIR_GI_MAX_MERGE_W;
        }
        if (w_dst > 0.0f) wSum += w_dst;
    }

    // ── Peer terms: m_peer(y_i) per Eq. 38 (defensive variant) ──────────
    for (uint32_t i = 0; i < numPeers; i++) {
        const GIReservoir& src = peers[i];
        if (!src.valid || src.M <= 0.0f) continue;
        if (!giShiftConnectable(dstSurf.position, dstSurf.normal,
                                 dstSurf.roughness, src)) continue;

        float3 wi;
        GIReservoir tmp = src;
        float pHatYi = giEvalTargetPdf(dstSurf, tmp, wi);   // p̂(y_i) at dst
        if (!(pHatYi > 0.0f)) continue;

        float jac = giJacobian(dstSurf.position, src.visiblePos, src);
        if (!(jac > 0.0f)) continue;

        // p̂_{←i}(y_i) at dst = p̂_i(T_i^{-1}(y_i)) · |∂T_i^{-1}/∂y_i|
        // For the reconnection shift, T_i^{-1}(y_i) puts the same sample
        // (x_r, n_r, L_o) back at peer i's surface (where it originally
        // came from). So p̂_i evaluated there equals the value the peer
        // had at the time it streamed the sample, which is exactly what
        // we'd compute as giEvalTargetPdf(peerSurf[i], src). The reverse
        // Jacobian is 1/jac.
        float3 wi_at_i;
        GIReservoir tmp2 = src;
        float pHatAtI = giEvalTargetPdf(peerSurf[i], tmp2, wi_at_i);
        if (!(pHatAtI > 0.0f)) continue;
        // c_{←i}(y_i) = pHatAtI / jac    (since reverse Jac is 1/jac)
        float c_li = pHatAtI / jac;

        float denom = Mc * pHatYi + Mnon * c_li;
        if (!(denom > 0.0f)) continue;

        float m_i = (Mnon * invM) * (c_li / denom);

        // Eq. 19: w_i = m_i · p̂(y_i) · W_i · jac
        float w_i = m_i * pHatYi * src.W * jac;
        if (!isfinite(w_i) || !(w_i > 0.0f)) continue;
        // Same bound as the held term — a peer with an outsized W·M·jac
        // product would otherwise hijack the destination's reservoir.
        if (w_i > RESTIR_GI_MAX_MERGE_W) w_i = RESTIR_GI_MAX_MERGE_W;

        wSum += w_i;

        // Reservoir resampling step. Use peer-specific u01 for proper
        // marginal probabilities under streaming RIS.
        float u = (u01s) ? u01s[i] : 0.5f;
        if (u * wSum < w_i) {
            dst.visiblePos     = dstSurf.position;
            dst.visibleNormal  = dstSurf.normal;
            dst.samplePos      = src.samplePos;
            dst.sampleNormal   = src.sampleNormal;
            dst.sampleRadiance = src.sampleRadiance;
            dst.pHat           = pHatYi;
            gris_cHat(dst)     = pHatYi;     // becomes canonical at dst
            dst.xrRoughness    = src.xrRoughness;
            dst.isEnv          = src.isEnv;
            dst.valid          = 1u;
        }
    }

    // Update confidence sum, finalise W via Eq. 22.
    dst.M = M;
    if (dst.valid && dst.pHat > 0.0f) {
        float W = wSum / dst.pHat;
        if (W > RESTIR_GI_MAX_W) W = RESTIR_GI_MAX_W;
        if (!isfinite(W)) W = 0.0f;
        dst.W = W;
    } else {
        dst.W = 0.0f;
    }
}

// Convenience: cap dst's confidence at `mCap` (paper §6.4 — bounds temporal
// correlation length so cross-frame `b_k` decays geometrically).
__device__ inline void gris_capM(GIReservoir& r, float mCap) {
    if (r.M > mCap) r.M = mCap;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward-compatible shims for existing call sites in ReSTIRGI.cu and the
// OptiX raygens. They forward to the new primitives so the GI pipeline
// continues to work without touching every call site.
// ─────────────────────────────────────────────────────────────────────────────

// Old name kept for source compatibility — used during initial-candidate
// generation. Caller MUST also call giReservoirFinalize (defined to forward
// to gris_finalizeInitial when M > 0 and we know it's a single-stream init).
__device__ inline bool giReservoirUpdate(
    GIReservoir& r, float& wSum,
    const float3& visiblePos, const float3& visibleNormal,
    bool isEnv, const float3& samplePos, const float3& sampleNormal,
    const float3& sampleRadiance,
    float pHat, float wCandidate, float u01)
{
    if (pHat <= 0.0f || wCandidate <= 0.0f) {
        r.M += 1.0f;
        return false;
    }
    float pSrc = pHat / wCandidate;
    return gris_streamCandidate(r, wSum,
                                visiblePos, visibleNormal,
                                isEnv, samplePos, sampleNormal, sampleRadiance,
                                pHat, pSrc, u01);
}

// (Legacy giReservoirCombine 5-arg variant has been removed: Lin et al.
// pairwise MIS needs the source's surface so peer.cHat^src can be computed
// at the destination measure. Callers MUST migrate to gris_mergeMultiPair,
// which takes parallel reservoir + surface arrays.)
