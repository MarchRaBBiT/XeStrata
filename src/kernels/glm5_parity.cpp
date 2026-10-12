// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/glm5_parity.cpp - GLM-5.3-Flash's kernels (glm5.hpp) against host references written from llama.cpp
// f9a95ae's graph and ggml CPU operators (MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE): the mHC read, write and mean, the sigmoid router, the KDA gate and step, the clamped
// SwiGLU.  The references are in double where ggml sums in float, so the tolerances are those of float rounding.
#include "strata/kernels/glm5.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int HC = 4, MIX = 24;
std::mt19937 rng(5353);

std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int compare(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double worst = 0;
    long long bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]) / std::max(1.0, std::fabs(want[i]));
        worst = std::max(worst, err);
        if (!(err <= tol)) ++bad;
    }
    std::printf("  %-34s worst %.2e  %s\n", what, worst, bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}

// ---- mHC
void ref_hc_pre(const std::vector<float>& x, const std::vector<float>& mixes, const float* scale, const float* base,
                double rms_eps, double hc_eps, int iters, int64_t N, int64_t T, std::vector<double>& y,
                std::vector<double>& post, std::vector<double>& comb) {
    y.assign((size_t) (T * N), 0);
    post.assign((size_t) (T * HC), 0);
    comb.assign((size_t) (T * HC * HC), 0);
    for (int64_t t = 0; t < T; ++t) {
        double ss = 0;
        for (int64_t i = 0; i < HC * N; ++i) ss += (double) x[(size_t) (t * HC * N + i)] * x[(size_t) (t * HC * N + i)];
        const double inv = 1.0 / std::sqrt(ss / (double) (HC * N) + rms_eps);
        double m[MIX];
        for (int i = 0; i < MIX; ++i) m[i] = mixes[(size_t) (t * MIX + i)] * inv;
        double pre[HC];
        for (int s = 0; s < HC; ++s) {
            pre[s] = sigm(m[s] * scale[0] + base[s]) + hc_eps;
            post[(size_t) (t * HC + s)] = 2.0 * sigm(m[HC + s] * scale[1] + base[HC + s]);
        }
        double c[HC * HC];
        for (int src = 0; src < HC; ++src) {
            double mx = -INFINITY, sum = 0;
            for (int dst = 0; dst < HC; ++dst) mx = std::max(mx, m[2 * HC + dst + HC * src] * scale[2] + base[2 * HC + dst + HC * src]);
            for (int dst = 0; dst < HC; ++dst) {
                c[dst + HC * src] = std::exp(m[2 * HC + dst + HC * src] * scale[2] + base[2 * HC + dst + HC * src] - mx);
                sum += c[dst + HC * src];
            }
            for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] = c[dst + HC * src] / sum + hc_eps;
        }
        auto cols = [&] {
            for (int dst = 0; dst < HC; ++dst) {
                double sum = hc_eps;
                for (int src = 0; src < HC; ++src) sum += c[dst + HC * src];
                for (int src = 0; src < HC; ++src) c[dst + HC * src] /= sum;
            }
        };
        auto rows = [&] {
            for (int src = 0; src < HC; ++src) {
                double sum = hc_eps;
                for (int dst = 0; dst < HC; ++dst) sum += c[dst + HC * src];
                for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] /= sum;
            }
        };
        cols();
        for (int i = 1; i < iters; ++i) { rows(); cols(); }
        for (int i = 0; i < HC * HC; ++i) comb[(size_t) (t * HC * HC + i)] = c[i];
        for (int64_t n = 0; n < N; ++n) {
            double v = 0;
            for (int s = 0; s < HC; ++s) v += x[(size_t) ((t * HC + s) * N + n)] * pre[s];
            y[(size_t) (t * N + n)] = v;
        }
    }
}

int test_hc() {
    const int64_t N = 4096, T = 3;
    const float scale[3] = {0.7f, 1.3f, 2.1f};
    std::vector<float> base = randn(MIX, 0.5f), x = randn((size_t) (T * HC * N)), mixes = randn((size_t) (T * MIX), 30.0f);
    const float rms_eps = 1e-5f, hc_eps = 1e-6f;
    std::vector<double> ry, rpost, rcomb;
    ref_hc_pre(x, mixes, scale, base.data(), rms_eps, hc_eps, 20, N, T, ry, rpost, rcomb);

    strata::parity::Device dev;
    std::vector<float> sc(scale, scale + 3);
    float *dx = dev.upload(x), *dm = dev.upload(mixes), *ds = dev.upload(sc), *db = dev.upload(base);
    float *dy = dev.alloc<float>((size_t) (T * N)), *dp = dev.alloc<float>((size_t) (T * HC)),
          *dc = dev.alloc<float>((size_t) (T * HC * HC));
    strata::kernels::glm5_hc_pre(dx, dm, ds, db, rms_eps, hc_eps, 20, dy, dp, dc, N, T, nullptr);
    std::vector<float> y((size_t) (T * N)), post((size_t) (T * HC)), comb((size_t) (T * HC * HC));
    strata::parity::copy_out(y.data(), dy, y.size());
    strata::parity::copy_out(post.data(), dp, post.size());
    strata::parity::copy_out(comb.data(), dc, comb.size());
    int bad = compare("hc_pre y", y, ry, 1e-4) + compare("hc_pre post", post, rpost, 1e-5) +
              compare("hc_pre comb (Sinkhorn 20)", comb, rcomb, 1e-4);
    // Sinkhorn: the columns sum to ~1 after the last round
    double worst_col = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int dst = 0; dst < HC; ++dst) {
            double s = 0;
            for (int src = 0; src < HC; ++src) s += comb[(size_t) (t * 16 + dst + HC * src)];
            worst_col = std::max(worst_col, std::fabs(s - 1.0));
        }
    std::printf("  %-34s %.2e  %s\n", "hc_pre comb column sums - 1", worst_col, worst_col < 1e-4 ? "ok" : "*** FAIL ***");
    bad += worst_col < 1e-4 ? 0 : 1;

    // the write, out of place and in place on the residual
    std::vector<float> f = randn((size_t) (T * N)), res = randn((size_t) (T * HC * N));
    std::vector<double> rout((size_t) (T * HC * N));
    for (int64_t t = 0; t < T; ++t)
        for (int d = 0; d < HC; ++d)
            for (int64_t n = 0; n < N; ++n) {
                double v = (double) f[(size_t) (t * N + n)] * post[(size_t) (t * HC + d)];
                for (int s = 0; s < HC; ++s)
                    v += (double) res[(size_t) ((t * HC + s) * N + n)] * comb[(size_t) (t * 16 + d + HC * s)];
                rout[(size_t) ((t * HC + d) * N + n)] = v;
            }
    float *df = dev.upload(f), *dr = dev.upload(res), *dout = dev.alloc<float>(res.size());
    strata::kernels::glm5_hc_post(df, dr, dp, dc, dout, N, T, nullptr);
    std::vector<float> out(res.size());
    strata::parity::copy_out(out.data(), dout, out.size());
    bad += compare("hc_post", out, rout, 1e-5);
    strata::kernels::glm5_hc_post(df, dr, dp, dc, dr, N, T, nullptr);
    strata::parity::copy_out(out.data(), dr, out.size());
    bad += compare("hc_post in place", out, rout, 1e-5);

    std::vector<double> rmean((size_t) (T * N));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            double v = 0;
            for (int s = 0; s < HC; ++s) v += x[(size_t) ((t * HC + s) * N + n)];
            rmean[(size_t) (t * N + n)] = v / HC;
        }
    float* dmean = dev.alloc<float>((size_t) (T * N));
    strata::kernels::glm5_hc_mean(dx, dmean, N, T, nullptr);
    std::vector<float> mean((size_t) (T * N));
    strata::parity::copy_out(mean.data(), dmean, mean.size());
    bad += compare("hc_mean", mean, rmean, 1e-6);
    return bad;
}

// ---- the sigmoid router
int test_router() {
    const int E = 288, K = 8, T = 40;
    std::vector<float> logits = randn((size_t) T * E, 2.0f), bias = randn(E, 0.05f);
    for (int e = 0; e < 12; ++e) logits[(size_t) (3 * E + e)] = 1.25f;   // token 3: a 12-way tie (with bias 0 below)
    for (int e = 0; e < 12; ++e) bias[(size_t) e] = 0.0f;
    std::vector<int> rid((size_t) T * K);
    std::vector<double> rw((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        // the kernel's order in float: the larger biased score, then the lower index
        std::vector<float> sel(E);
        for (int e = 0; e < E; ++e)
            sel[(size_t) e] = 1.0f / (1.0f + std::exp(-logits[(size_t) (t * E + e)])) + bias[(size_t) e];
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
        double sum = 0;
        for (int i = 0; i < K; ++i) {
            rid[(size_t) (t * K + i)] = idx[(size_t) i];
            rw[(size_t) (t * K + i)] = sigm(logits[(size_t) (t * E + idx[(size_t) i])]);
            sum += rw[(size_t) (t * K + i)];
        }
        sum = std::max(sum, 6.103515625e-05);
        for (int i = 0; i < K; ++i) rw[(size_t) (t * K + i)] = rw[(size_t) (t * K + i)] / sum * 2.5;
    }
    strata::parity::Device dev;
    float *dl = dev.upload(logits), *db = dev.upload(bias), *dw = dev.alloc<float>((size_t) T * K);
    int32_t* di = dev.alloc<int32_t>((size_t) T * K);
    strata::kernels::glm5_router_sigmoid(dl, db, T, E, K, true, 2.5f, di, dw, nullptr);
    std::vector<int32_t> id((size_t) T * K);
    std::vector<float> w((size_t) T * K);
    strata::parity::copy_out(id.data(), di, id.size());
    strata::parity::copy_out(w.data(), dw, w.size());
    long long id_bad = 0;
    for (size_t i = 0; i < id.size(); ++i) id_bad += id[i] != rid[i];
    std::printf("  %-34s %lld wrong  %s\n", "router ids (288 -> 8, a 12-way tie)", id_bad, id_bad ? "*** FAIL ***" : "ok");
    return (id_bad ? 1 : 0) + compare("router weights (norm, x2.5)", w, rw, 1e-5);
}

// ---- KDA
int test_kda() {
    const int H = 4, S = 128, T = 3;
    const float lb = -5.0f;
    std::vector<float> f = randn((size_t) T * H * S), dt = randn((size_t) H * S, 0.5f), a(H);
    for (int h = 0; h < H; ++h) a[(size_t) h] = -std::exp(0.3f * (float) h - 0.5f);
    std::vector<double> rg((size_t) T * H * S);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s) {
                const size_t i = (size_t) ((t * H + h) * S + s);
                rg[i] = lb * sigm(-((double) f[i] + dt[(size_t) (h * S + s)]) * a[(size_t) h]);
            }
    strata::parity::Device dev;
    float *dfa = dev.upload(f), *ddt = dev.upload(dt), *da = dev.upload(a), *dg = dev.alloc<float>(f.size());
    strata::kernels::glm5_kda_gate(dfa, ddt, da, lb, dg, H, S, T, nullptr);
    std::vector<float> g(f.size());
    strata::parity::copy_out(g.data(), dg, g.size());
    int bad = compare("kda_gate", g, rg, 1e-5);

    // T steps from a random state; q, k L2-normalised per head
    std::vector<float> state = randn((size_t) S * H * S, 0.1f);
    std::vector<double> rs(state.begin(), state.end());
    float* dstate = dev.upload(state);
    std::vector<double> rout_all;
    std::vector<float> out_all;
    for (int t = 0; t < T; ++t) {
        std::vector<float> q = randn((size_t) H * S), k = randn((size_t) H * S), v = randn((size_t) H * S), beta(H);
        for (auto* p : {&q, &k})
            for (int h = 0; h < H; ++h) {
                double n = 0;
                for (int i = 0; i < S; ++i) n += (double) (*p)[(size_t) (h * S + i)] * (*p)[(size_t) (h * S + i)];
                for (int i = 0; i < S; ++i) (*p)[(size_t) (h * S + i)] = (float) ((*p)[(size_t) (h * S + i)] / std::sqrt(n + 1e-6));
            }
        for (int h = 0; h < H; ++h) beta[(size_t) h] = (float) sigm(0.3 * h - 0.2);
        const float* gt = g.data() + (size_t) t * H * S;
        for (int h = 0; h < H; ++h) {
            auto at = [&](int i, int j) -> double& { return rs[((size_t) i * H + h) * S + j]; };
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j) at(i, j) *= std::exp((double) gt[h * S + i]);
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += at(i, j) * k[(size_t) (h * S + i)];
                const double delta = ((double) v[(size_t) (h * S + j)] - kv) * beta[(size_t) h];
                for (int i = 0; i < S; ++i) at(i, j) += (double) k[(size_t) (h * S + i)] * delta;
            }
            for (int j = 0; j < S; ++j) {
                double o = 0;
                for (int i = 0; i < S; ++i) o += at(i, j) * q[(size_t) (h * S + i)];
                rout_all.push_back(o / std::sqrt((double) S));
            }
        }
        float *dq = dev.upload(q), *dk = dev.upload(k), *dv = dev.upload(v), *dbeta = dev.upload(beta),
              *dout = dev.alloc<float>((size_t) H * S);
        strata::kernels::glm5_kda_step(dstate, dq, dk, dv, dg + (size_t) t * H * S, dbeta, dout, H, nullptr);
        strata::parity::sync();
        std::vector<float> out((size_t) H * S);
        strata::parity::copy_out(out.data(), dout, out.size());
        out_all.insert(out_all.end(), out.begin(), out.end());
    }
    strata::parity::copy_out(state.data(), dstate, state.size());
    bad += compare("kda_step out (3 steps)", out_all, rout_all, 1e-4);
    bad += compare("kda_step state", state, rs, 1e-4);
    return bad;
}

// ---- the clamped SwiGLU
int test_swiglu_clamp() {
    const size_t n = 4096;
    std::vector<float> g = randn(n, 8.0f), u = randn(n, 8.0f);
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) {
        const double gv = std::min((double) g[i], 10.0), uv = std::clamp((double) u[i], -10.0, 10.0);
        r[i] = gv / (1.0 + std::exp(-gv)) * uv;
    }
    strata::parity::Device dev;
    float *dg = dev.upload(g), *du = dev.upload(u), *dout = dev.alloc<float>(n);
    strata::kernels::glm5_swiglu_clamp(dg, du, dout, (int64_t) n, 10.0f, nullptr);
    std::vector<float> out(n);
    strata::parity::copy_out(out.data(), dout, n);
    int bad = compare("swiglu_clamp (limit 10)", out, r, 1e-5);
    strata::kernels::glm5_swiglu_clamp(dg, du, dg, (int64_t) n, 10.0f, nullptr);
    strata::parity::copy_out(out.data(), dg, n);
    return bad + compare("swiglu_clamp in place", out, r, 1e-5);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: glm5_parity [--selftest]\n");
            return 2;
        }
    std::printf("glm5_parity: GLM-5.3-Flash's kernels against host references\n");
    const int bad = test_hc() + test_router() + test_kda() + test_swiglu_clamp();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
