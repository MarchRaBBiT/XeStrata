// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/glm5.hpp - GLM-5.3-Flash's (glm5-next) own pieces: the mHC streams, the sigmoid router, the
// KDA gate and step, and the clamped SwiGLU.  The arithmetic follows llama.cpp's src/models/glm5-next.cpp and the ggml
// CPU operators it uses (ggml_dsv4_hc_comb, ggml_gated_delta_net with a per-channel gate, ggml_swiglu_clamp) at
// f9a95ae; MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.
//
// Every buffer is a device pointer; `stream` null runs synchronously.  T is the number of tokens, N the model width.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The mHC streams' read before a block (llama.cpp's build_hc_pre), for the hc = 4 streams.
///   x      [T][4][N]   the streams
///   mixes  [T][24]     hc_fn applied to the token's 4N stream values as they are: their RMS normalisation (eps
///                      `rms_eps`, no weight) is a factor of the product and is applied here
///   scale  [3], base [24]   hc_*_scale and hc_*_base
/// Out: y [T][N] = sum_s pre[s] x[s], pre = sigmoid(mix[0..4) scale[0] + base[0..4)) + hc_eps;
///      post [T][4] = 2 sigmoid(mix[4..8) scale[1] + base[4..8));
///      comb [T][16], comb[dst + 4 src]: mix[8..24) scale[2] + base[8..24), softmax over dst, + hc_eps, then the
///      columns (sum over src) normalised, then `iters` - 1 rounds of rows (sum over dst) and columns, each sum + hc_eps.
void glm5_hc_pre(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps, float hc_eps,
                 int iters, float* y, float* post, float* comb, int64_t n_embd, int64_t n_tokens, void* stream);

/// The mHC streams' write after a block (llama.cpp's build_hc_post): out[t][d] = f[t] post[t][d] + sum_s
/// residual[t][s] comb[t][d + 4 s].  `out` may be `residual` (each element's four streams are read before written).
void glm5_hc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
                  int64_t n_embd, int64_t n_tokens, void* stream);

/// The streams' mean before the output norm: y[t] = (x[t][0] + x[t][1] + x[t][2] + x[t][3]) / 4.
void glm5_hc_mean(const float* x, float* y, int64_t n_embd, int64_t n_tokens, void* stream);

/// The sigmoid router (llama.cpp's build_moe_ffn with gating SIGMOID and exp_probs_b): p = sigmoid(logits); the k
/// experts with the largest p + bias (ties to the lower index), weights p of those, divided by their sum (clamped
/// below at 2**-14) when `norm`, times `scale`.  logits [T][E], bias [E]; ids, weights [T][k].  k <= 32.
void glm5_router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm,
                         float scale, int32_t* ids, float* weights, void* stream);

/// KDA's per-channel decay (llama.cpp's build_kda_layer with a gate lower bound): g = lb sigmoid(-(f + dt_bias) a),
/// a = ssm_a (it holds -exp(A_log)) per head.  f, g [T][H][S]; dt_bias [H S]; ssm_a [H].  lb finite and below 0.
void glm5_kda_gate(const float* f, const float* dt_bias, const float* ssm_a, float lb, float* g, int64_t n_heads,
                   int64_t head_dim, int64_t n_tokens, void* stream);

/// One KDA step (ggml_gated_delta_net with a gate per key channel), IN PLACE on `state`, for S = 128:
///   state (S, H, S) f32 as Strata's GDN state, state[(i H + h) S + j] = S_h[i][j] (i the key channel)
///   q, k  (H, S) L2-normalised, q NOT scaled (1/sqrt(S) is applied to the readout)
///   v     (H, S);  g (H, S) the PRE-exp decay of each key channel;  beta (H,) after its sigmoid
///   out   (H, S) = (S_h' q) / sqrt(S), S_h' = diag(exp g) S_h + k delta^T, delta = (v - (diag(exp g) S_h)^T k) beta
void glm5_kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, void* stream);

/// GLM's clamped SwiGLU (ggml_swiglu_clamp): out = silu(min(gate, limit)) clamp(up, -limit, limit), n values.
/// `out` may be `gate` or `up`.
void glm5_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream);

}  // namespace strata::kernels
