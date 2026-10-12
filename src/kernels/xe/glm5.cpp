// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/glm5.cpp - see include/strata/kernels/glm5.hpp.
//
// Transcribed from llama.cpp f9a95ae (src/models/glm5-next.cpp; ggml/src/ggml-cpu/ops.cpp's dsv4_hc_comb,
// gated_delta_net and swiglu_clamp); MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE.  The KDA step keeps the layout and the sub-group shape of native_gdn.cpp.
#include "strata/kernels/glm5.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_glm5
#include "strata/kernels/device_arch_tag.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HC = 4;                  // the mHC streams (model_spec refuses another count)
constexpr int MIX = (2 + HC) * HC;     // pre, post and the HC x HC comb
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

void glm5_hc_pre(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps, float hc_eps,
                 int iters, float* y, float* post, float* comb, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_embd <= 0 || iters < 1) throw core::DeviceError("glm5_hc_pre: n_embd " + std::to_string(n_embd) + ", iters " +
                                                          std::to_string(iters));
    constexpr int WG = 256;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> pre_l(sycl::range<1>(HC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* xt = x + t * HC * N;
            // the RMS of the token's 4N stream values
            float ss = 0.0f;
            for (int64_t i = lid; i < HC * N; i += WG) ss += xt[i] * xt[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            if (lid == 0) {
                const float inv = 1.0f / sycl::sqrt(ss / (float) (HC * N) + rms_eps);
                float m[MIX];
                for (int i = 0; i < MIX; ++i) m[i] = mixes[t * MIX + i] * inv;
                for (int s = 0; s < HC; ++s) {
                    pre_l[s] = sigmoid(m[s] * scale[0] + base[s]) + hc_eps;
                    post[t * HC + s] = 2.0f * sigmoid(m[HC + s] * scale[1] + base[HC + s]);
                }
                // ggml_dsv4_hc_comb: softmax over dst for each src, + eps, then the Sinkhorn rounds
                float c[HC * HC];
                for (int src = 0; src < HC; ++src) {
                    float mx = -INFINITY;
                    for (int dst = 0; dst < HC; ++dst) {
                        const int idx = dst + HC * src;
                        c[idx] = m[2 * HC + idx] * scale[2] + base[2 * HC + idx];
                        mx = sycl::fmax(mx, c[idx]);
                    }
                    float sum = 0.0f;
                    for (int dst = 0; dst < HC; ++dst) {
                        const int idx = dst + HC * src;
                        c[idx] = sycl::exp(c[idx] - mx);
                        sum += c[idx];
                    }
                    const float inv_sum = 1.0f / sum;
                    for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] = c[dst + HC * src] * inv_sum + hc_eps;
                }
                auto norm_cols = [&]() {
                    for (int dst = 0; dst < HC; ++dst) {
                        float sum = hc_eps;
                        for (int src = 0; src < HC; ++src) sum += c[dst + HC * src];
                        const float inv_sum = 1.0f / sum;
                        for (int src = 0; src < HC; ++src) c[dst + HC * src] *= inv_sum;
                    }
                };
                auto norm_rows = [&]() {
                    for (int src = 0; src < HC; ++src) {
                        float sum = hc_eps;
                        for (int dst = 0; dst < HC; ++dst) sum += c[dst + HC * src];
                        const float inv_sum = 1.0f / sum;
                        for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] *= inv_sum;
                    }
                };
                norm_cols();
                for (int i = 1; i < iters; ++i) {
                    norm_rows();
                    norm_cols();
                }
                for (int i = 0; i < HC * HC; ++i) comb[t * HC * HC + i] = c[i];
            }
            sycl::group_barrier(it.get_group());
            const float p0 = pre_l[0], p1 = pre_l[1], p2 = pre_l[2], p3 = pre_l[3];
            for (int64_t n = lid; n < N; n += WG)   // ggml: x0 p0, then + x1 p1, + x2 p2, + x3 p3
                y[t * N + n] = ((xt[n] * p0 + xt[N + n] * p1) + xt[2 * N + n] * p2) + xt[3 * N + n] * p3;
        });
    });
    sync_if_needed(stream, event, "glm5_hc_pre");
}

void glm5_hc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
                  int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        const float fv = f[i];
        float r[HC];
        for (int s = 0; s < HC; ++s) r[s] = residual[(t * HC + s) * N + n];
        for (int d = 0; d < HC; ++d) {
            float v = fv * post[t * HC + d];
            for (int s = 0; s < HC; ++s) v += r[s] * comb[t * HC * HC + d + HC * s];
            out[(t * HC + d) * N + n] = v;
        }
    });
    sync_if_needed(stream, event, "glm5_hc_post");
}

void glm5_hc_mean(const float* x, float* y, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        const float* xt = x + t * HC * N + n;
        y[i] = (((xt[0] + xt[N]) + xt[2 * N]) + xt[3 * N]) * (1.0f / HC);
    });
    sync_if_needed(stream, event, "glm5_hc_mean");
}

void glm5_router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm,
                         float scale, int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0) return;
    if (n_expert <= 0 || k <= 0 || k > WARP || k > n_expert)
        throw core::DeviceError("glm5_router_sigmoid: " + std::to_string(k) + " of " + std::to_string(n_expert) +
                                " experts (at most 32 picks)");
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), lane = (int) sg.get_local_linear_id();
            const float* l = logits + (size_t) t * n_expert;
            // pass i: the best below pass i - 1's pick in the order (larger biased score, then lower index)
            float pv = INFINITY, mine = 0.0f;
            int pi = -1;
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;
                for (int e = lane; e < n_expert; e += WARP) {
                    const float v = sigmoid(l[e]) + bias[e];
                    if ((v < pv || (v == pv && e > pi)) && (v > bv || (v == bv && e < bi))) { bv = v; bi = e; }
                }
                for (int off = WARP / 2; off > 0; off >>= 1) {
                    const float ov = sycl::permute_group_by_xor(sg, bv, off);
                    const int oi = sycl::permute_group_by_xor(sg, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == i) {
                    mine = sigmoid(l[bi]);
                    ids[(size_t) t * k + i] = bi;
                }
                pv = bv;
                pi = bi;
            }
            float div = 1.0f;
            if (norm) {
                const float sum = sycl::reduce_over_group(sg, lane < k ? mine : 0.0f, sycl::plus<float>());
                div = sycl::fmax(sum, 6.103515625e-05f);   // 2**-14, as llama.cpp
            }
            if (lane < k) weights[(size_t) t * k + lane] = mine / div * scale;
        });
    sync_if_needed(stream, event, "glm5_router_sigmoid");
}

void glm5_kda_gate(const float* f, const float* dt_bias, const float* ssm_a, float lb, float* g, int64_t n_heads,
                   int64_t head_dim, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || head_dim <= 0 || !(lb < 0.0f) || !std::isfinite(lb))
        throw core::DeviceError("glm5_kda_gate: a finite negative lower bound and positive head shapes");
    const int64_t C = n_heads * head_dim;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, h = c / head_dim;
        const float x = (f[i] + dt_bias[c]) * ssm_a[h];
        g[i] = lb * sigmoid(-x);
    });
    sync_if_needed(stream, event, "glm5_kda_gate");
}

void glm5_kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, void* stream) {
    constexpr int S = 128;
    constexpr int COLS = 4;   // columns (sub-groups) a work-group
    if (n_heads <= 0 || n_heads > 65535) throw core::DeviceError("glm5_kda_step: " + std::to_string(n_heads) + " heads");
    const int H = (int) n_heads;
    const float scale = 1.0f / sycl::sqrt((float) S);
    // one sub-group a (head, column): its four state rows a lane in registers, as native_gdn_step
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) H, (size_t) S * WARP}, {1, (size_t) COLS * WARP}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id();
            const int col = (int) it.get_group(1) * COLS + (int) sg.get_group_linear_id();
            float s[4], kr[4], qr[4], dr[4];
            float kv = 0.0f;
            for (int r = 0; r < 4; ++r) {
                const int i = r * WARP + lane;
                dr[r] = sycl::exp(g[head * S + i]);
                s[r] = dr[r] * state[((size_t) i * H + head) * S + col];   // the decay first, as ggml
                kr[r] = k[head * S + i];
                qr[r] = q[head * S + i];
                kv += s[r] * kr[r];
            }
            kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
            const float delta = (v[head * S + col] - kv) * beta[head];
            float o = 0.0f;
            for (int r = 0; r < 4; ++r) {
                s[r] += kr[r] * delta;
                o += s[r] * qr[r];
            }
            o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
            if (lane == 0) out[head * S + col] = o * scale;
            for (int r = 0; r < 4; ++r) state[((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
        });
    sync_if_needed(stream, event, "glm5_kda_step");
}

void glm5_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float gv = sycl::fmin(gate[i], limit);
        const float uv = sycl::clamp(up[i], -limit, limit);
        out[i] = gv / (1.0f + sycl::exp(-gv)) * uv;
    });
    sync_if_needed(stream, event, "glm5_swiglu_clamp");
}

}  // namespace strata::kernels
