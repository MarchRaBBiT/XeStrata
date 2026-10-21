// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_dit.cpp - see include/strata/image/mf_dit.hpp.

#include "strata/image/mf_dit.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strata::image {

namespace gpu = strata::gpu;

// The kernels' helpers in a namespace of this file's own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1), so a helper of the same
// name and signature in another image file (qi_vae.cpp's to_rgba) gave its kernel to both.
namespace mf_dit_detail {

namespace syclex = sycl::ext::oneapi::experimental;
using strata::kernels::f16_from_f32;

constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeBF16 = 30;
constexpr float kEps = 1e-6f;

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Mage-Flow DiT): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (Mage-Flow DiT): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

float bf16_value(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x rounded to the nearest bf16 (ties to even), as a float
float round_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// row of K bf16 values -> Q8_0 (ggml's reference rounding): blocks of an FP16 scale and 32 int8 values
void quantize_row(const uint16_t* src, int64_t K, uint8_t* dst) {
    for (int64_t b = 0; b < K / 32; ++b) {
        float v[32], amax = 0.f;
        for (int j = 0; j < 32; ++j) {
            v[j] = bf16_value(src[b * 32 + j]);
            amax = std::max(amax, std::fabs(v[j]));
        }
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        const uint16_t dh = f16_from_f32(d);
        int8_t* qs = reinterpret_cast<int8_t*>(dst + b * 34 + 2);
        for (int j = 0; j < 32; ++j) qs[j] = (int8_t) std::lround(v[j] * id);
        std::memcpy(dst + b * 34, &dh, 2);
    }
}

// the rows of a [n][k] bf16 matrix quantized on the host's threads
void quantize_matrix(const uint16_t* src, int64_t n, int64_t k, uint8_t* dst) {
    const uint64_t row = (uint64_t) (k / 32 * 34);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([=] {
            for (int64_t r = t; r < n; r += nt) quantize_row(src + r * k, k, dst + r * row);
        });
    for (auto& x : th) x.join();
}

// Mage-Flow's activations pass FP16's range (its model ran in bf16: the image stream's normed rows up to 1.9e5, the
// attention's v up to 6.1e5, a text MLP's output 5.7e6 at the first steps of a 1024 x 1024 Turbo run).  So a product's
// FP16 input holds each row divided by a power of two (rs, at least 1) that brings its largest value under kRowLim, and
// the product's output rows are multiplied back by it; the attention's v is divided by one power of two a head over a
// context's rows (its output is a weighted sum of them), and the output multiplied back.  Powers of two leave the
// rounding as it is; FP16 keeps 3 more bits than bf16.
constexpr float kRowLim = 16384.f;

inline float pow2_scale(float amax) {
    return amax > kRowLim ? sycl::exp2(sycl::ceil(sycl::log2(amax / kRowLim))) : 1.f;
}

bool has_sg32(sycl::queue& q) {
    static core::PerDevice<bool> sg32;
    return sg32.get(q.get_device(), [&q] {
        const auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        return std::find(sizes.begin(), sizes.end(), (size_t) 32) != sizes.end();
    });
}

// A row-wise kernel writing FP16 with each row's scale: Op (LnMod, Gelu, Heads below) gives a row's statistics and its
// values; a sub-group of 32 a row, 8 rows a work-group (or, without sub-groups of 32, a work-item a row).
constexpr int64_t kSG = 32, kRows = 8;

template<class Op> struct Rows16 {
    Op op;
    int64_t T, C;
    uint16_t* y16;
    float* rs;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t t = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (t >= T) return;
        const auto sum = [&](float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, kSG, sum);
        float m = 0.f;
        for (int64_t c = lane; c < C; c += kSG) m = sycl::fmax(m, sycl::fabs(op.value(t, c, st)));
        const float sc = pow2_scale(sycl::reduce_over_group(sg, m, sycl::maximum<float>()));
        const float inv = 1.f / sc;
        for (int64_t c = lane; c < C; c += kSG) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        if (lane == 0) rs[t] = sc;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

template<class Op> void rows16(const Op& op, int64_t T, int64_t C, uint16_t* y16, float* rs, void* s) {
    if (T <= 0) return;
    sycl::queue& q = q_of(s);
    if (has_sg32(q)) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T + kRows - 1) / kRows * kRows * kSG)),
                                         sycl::range<1>((size_t) (kRows * kSG))),
                       Rows16<Op>{op, T, C, y16, rs});
        return;
    }
    q.parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> id) {
        const auto t = static_cast<int64_t>(id[0]);
        const auto st = op.stats(t, 0, 1, [](float v) { return v; });
        float m = 0.f;
        for (int64_t c = 0; c < C; ++c) m = sycl::fmax(m, sycl::fabs(op.value(t, c, st)));
        const float sc = pow2_scale(m), inv = 1.f / sc;
        for (int64_t c = 0; c < C; ++c) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        rs[t] = sc;
    });
}

struct Stats {
    float mean = 0.f, inv = 1.f;
};

// LayerNorm(x[t]) (no affine, eps 1e-6) * (1 + scale) + shift
struct LnMod {
    const float *x, *shift, *scale;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += r[c];
        const float mean = reduce(s) / (float) C;
        float v = 0.f;
        for (int64_t c = lane; c < C; c += step) v += (r[c] - mean) * (r[c] - mean);
        return {mean, 1.f / sycl::sqrt(reduce(v) / (float) C + kEps)};
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        return (x[t * C + c] - st.mean) * st.inv * (1.f + scale[c]) + shift[c];
    }
};

// GELU (tanh form) of x
struct Gelu {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float v = x[t * C + c];
        return 0.5f * v * (1.f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
};

// the attention's output o16 [T][C] times its heads' scales hs (head_dim channels a head)
struct Heads {
    const uint16_t* o16;
    const float* hs;
    int64_t C, hd;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return kernels::f32_from_f16(o16[t * C + c]) * hs[c / hd]; }
};

// part[r][h] = the largest |v| of row r's head h, rows of a ([na][C]) then b ([nb][C])
void head_max(const float* a, int64_t na, const float* b, int64_t nb, int64_t heads, int64_t hd, float* part, void* s) {
    const int64_t C = heads * hd;
    q_of(s).parallel_for(sycl::range<1>((size_t) ((na + nb) * heads)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / heads, h = i - r * heads;
        const float* row = (r < na ? a + r * C : b + (r - na) * C) + h * hd;
        float m = 0.f;
        for (int64_t d = 0; d < hd; ++d) m = sycl::fmax(m, sycl::fabs(row[d]));
        part[i] = m;
    });
}

// hs[h] = the scale of head h over the rows' part
void head_scale(const float* part, int64_t rows, int64_t heads, float* hs, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> id) {
        const auto h = static_cast<int64_t>(id[0]);
        float m = 0.f;
        for (int64_t r = 0; r < rows; ++r) m = sycl::fmax(m, part[r * heads + h]);
        hs[h] = pow2_scale(m);
    });
}

// For each row and head: q and k RMS-normed over the head (weights wq, wk; eps 1e-6) and rotated pairwise (2j, 2j + 1)
// by cos[t][j], sin[t][j] (rope false: not rotated), v divided by its head's scale; FP16 out.  A work-item a row's
// head.
void qk_rope_v16(const float* q, const float* k, const float* v, const float* wq, const float* wk, const float* cos,
                 const float* sin, bool rope, const float* hs, int64_t T, int64_t heads, int64_t hd, uint16_t* q16,
                 uint16_t* k16, uint16_t* v16, void* s) {
    if (T <= 0) return;
    const int64_t C = heads * hd, half = hd / 2;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * heads)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / heads, h = i - t * heads, o = t * C + h * hd;
        for (int which = 0; which < 2; ++which) {
            const float* src = (which == 0 ? q : k) + o;
            const float* w = which == 0 ? wq : wk;
            uint16_t* dst = (which == 0 ? q16 : k16) + o;
            float ss = 0.f;
            for (int64_t d = 0; d < hd; ++d) ss += src[d] * src[d];
            const float r = 1.f / sycl::sqrt(ss / (float) hd + kEps);
            for (int64_t j = 0; j < half; ++j) {
                const float a = src[2 * j] * r * w[2 * j], b = src[2 * j + 1] * r * w[2 * j + 1];
                const float c = rope ? cos[t * half + j] : 1.f, sn = rope ? sin[t * half + j] : 0.f;
                dst[2 * j] = f16_from_f32(a * c - b * sn);
                dst[2 * j + 1] = f16_from_f32(a * sn + b * c);
            }
        }
        const float inv = 1.f / hs[h];
        for (int64_t d = 0; d < hd; ++d) v16[o + d] = f16_from_f32(v[o + d] * inv);
    });
}

// y[t] = y[t] * rs[t] + b (rs null: 1; b null: none)
void rows_back(float* y, const float* rs, const float* b, int64_t T, int64_t N, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / N;
        float v = y[i];
        if (rs != nullptr) v *= rs[t];
        if (b != nullptr) v += b[i - t * N];
        y[i] = v;
    });
}

// x[t] += gate * h[t] (per channel)
void gate_add(float* x, const float* h, const float* gate, int64_t T, int64_t C, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        x[i] += gate[i % C] * h[i];
    });
}

void fill_i32(int32_t* p, int64_t n, int32_t v, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { p[id[0]] = v; });
}

// x += ds * v, v the guided velocity: out[0] (the prompt's) alone, or out[1] + cfg (out[0] - out[1])
void euler(float* x, const float* out, int64_t n, int ctxs, float cfg, float ds, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const float v = ctxs == 2 ? out[n + i] + cfg * (out[i] - out[n + i]) : out[i];
        x[i] += ds * v;
    });
}

}  // namespace mf_dit_detail

using namespace mf_dit_detail;

MfDit::MfDit() = default;

MfDit::~MfDit() {
    blocks_stream_.drop();
    for (void* p : allocs_) gpu::free(p);
}

bool MfDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8) {
    try {
        s_ = stream;
        dry_ = dry;
        q8_ = q8;
        SafeTensors st(path);
        file_ = std::make_unique<MappedFile>(path);
        const uint8_t* data = file_->data() + st.data_base();
        auto tensor = [&](const std::string& name) -> const SafeTensor& {
            const SafeTensor* t = st.find(name);
            if (t == nullptr) throw std::runtime_error("image engine (Mage-Flow DiT): no " + name);
            if (t->dtype != "BF16") throw std::runtime_error("image engine (Mage-Flow DiT): " + name + " is not bf16");
            return *t;
        };
        // the parts outside the blocks: FP16 weights and FP32 vectors on the GPU
        auto dense = [&](const std::string& name, const float** bias) {
            const SafeTensor& t = tensor(name + ".weight");
            W w;
            w.type = kTypeF16;
            w.n = t.shape[0];
            w.k = t.shape[1];
            w.row_bytes = (uint64_t) w.k * 2;
            weight_bytes_ += (uint64_t) (w.n * w.k) * 2 + (uint64_t) w.n * 4;
            if (!dry_) {
                const auto* src = reinterpret_cast<const uint16_t*>(data + t.begin);
                std::vector<uint16_t> h((size_t) (w.n * w.k));
                for (size_t i = 0; i < h.size(); ++i) h[i] = f16_from_f32(bf16_value(src[i]));
                uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
                gpu::copy(d, h.data(), h.size() * 2);
                w.d = d;
                const auto b = st.f32(name + ".bias");
                float* db = alloc<float>(allocs_, (int64_t) b.size());
                gpu::copy(db, b.data(), b.size() * 4);
                *bias = db;
            }
            return w;
        };
        img_in_ = dense("img_in", &b_img_in_);
        in_ch_ = img_in_.k;
        C_ = img_in_.n;
        txt_in_ = dense("txt_in", &b_txt_in_);
        txt_dim_ = txt_in_.k;
        t1_ = dense("time_text_embed.timestep_embedder.linear_1", &b_t1_);
        t2_ = dense("time_text_embed.timestep_embedder.linear_2", &b_t2_);
        nout_ = dense("norm_out.linear", &b_nout_);
        pout_ = dense("proj_out", &b_pout_);
        weight_bytes_ += (uint64_t) txt_dim_ * 4;
        if (!dry_) {
            const auto v = st.f32("txt_norm.weight");
            float* d = alloc<float>(allocs_, (int64_t) v.size());
            gpu::copy(d, v.data(), v.size() * 4);
            txt_norm_ = d;
        }
        hd_ = tensor("transformer_blocks.0.attn.norm_q.weight").shape[0];
        heads_ = C_ / hd_;
        int64_t axes_sum = 0;
        for (int64_t a : axes_) axes_sum += a;
        if (axes_sum != hd_) throw std::runtime_error("image engine (Mage-Flow DiT): the RoPE's axes are not the head");
        int64_t scratch = std::max({nout_.n * nout_.k, txt_in_.n * txt_in_.k, t2_.n * t2_.k});
        for (int i = 0;; ++i) {
            const std::string p = "transformer_blocks." + std::to_string(i) + ".";
            if (st.find(p + "img_mod.1.weight") == nullptr) break;
            Block b;
            struct Item {
                W* w;
                uint64_t* bias;
                std::string name;
            };
            const std::vector<Item> items = {
                {&b.imod, &b.b_imod, "img_mod.1"}, {&b.tmod, &b.b_tmod, "txt_mod.1"},
                {&b.q, &b.b_q, "attn.to_q"}, {&b.k, &b.b_k, "attn.to_k"}, {&b.v, &b.b_v, "attn.to_v"},
                {&b.tq, &b.b_tq, "attn.add_q_proj"}, {&b.tk, &b.b_tk, "attn.add_k_proj"},
                {&b.tv, &b.b_tv, "attn.add_v_proj"}, {&b.o, &b.b_o, "attn.to_out.0"},
                {&b.to, &b.b_to, "attn.to_add_out"}, {&b.imlp1, &b.b_imlp1, "img_mlp.net.0.proj"},
                {&b.imlp2, &b.b_imlp2, "img_mlp.net.2"}, {&b.tmlp1, &b.b_tmlp1, "txt_mlp.net.0.proj"},
                {&b.tmlp2, &b.b_tmlp2, "txt_mlp.net.2"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.nq, "attn.norm_q.weight"}, {&b.nk, "attn.norm_k.weight"},
                {&b.ntq, "attn.norm_added_q.weight"}, {&b.ntk, "attn.norm_added_k.weight"}};
            // the block's host bytes: the quantized weights (q8), then every FP32 vector
            uint64_t qbytes = 0, vbytes = 0;
            for (const Item& it : items) {
                const SafeTensor& t = tensor(p + it.name + ".weight");
                it.w->n = t.shape[0];
                it.w->k = t.shape[1];
                if (it.w->k % 32 != 0) throw std::runtime_error("image engine (Mage-Flow DiT): columns not 32s");
                it.w->type = q8_ ? kTypeQ8_0 : kTypeBF16;
                it.w->row_bytes = q8_ ? (uint64_t) (it.w->k / 32 * 34) : (uint64_t) it.w->k * 2;
                if (q8_) qbytes += it.w->row_bytes * (uint64_t) it.w->n;
                vbytes += (uint64_t) it.w->n * 4;
                scratch = std::max(scratch, it.w->n * it.w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) tensor(p + name).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan(items.size()), bspan(items.size()), nspan(norms.size());
            for (size_t j = 0; j < items.size(); ++j) {
                const SafeTensor& t = tensor(p + items[j].name + ".weight");
                const W& w = *items[j].w;
                const uint64_t bytes = w.row_bytes * (uint64_t) w.n;
                if (q8_) {
                    if (!dry_)
                        quantize_matrix(reinterpret_cast<const uint16_t*>(data + t.begin), w.n, w.k, host.data() + at);
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({data + t.begin, bytes, 0});
                }
                wspan[j] = spans.size() - 1;
            }
            auto vec = [&](const std::string& name, size_t* span) {
                const auto v = dry_ ? std::vector<float>((size_t) tensor(name).elements()) : st.f32(name);
                if (!dry_) std::memcpy(host.data() + at, v.data(), v.size() * 4);
                spans.push_back({host.data() + at, v.size() * 4, 0});
                at += v.size() * 4;
                *span = spans.size() - 1;
            };
            for (size_t j = 0; j < items.size(); ++j) vec(p + items[j].name + ".bias", &bspan[j]);
            for (size_t j = 0; j < norms.size(); ++j) vec(p + norms[j].second, &nspan[j]);
            const int layer = blocks_stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j) {
                items[j].w->d = reinterpret_cast<const void*>(blocks_stream_.offset(layer, (int) wspan[j]));
                *items[j].bias = blocks_stream_.offset(layer, (int) bspan[j]);
            }
            for (size_t j = 0; j < norms.size(); ++j) *norms[j].first = blocks_stream_.offset(layer, (int) nspan[j]);
            if (q8_ && !dry_) {   // the block's bf16 pages: read once, not kept in the page cache
                uint64_t lo = UINT64_MAX, hi = 0;
                for (const Item& it : items) {
                    const SafeTensor& t = tensor(p + it.name + ".weight");
                    lo = std::min(lo, st.data_base() + t.begin);
                    hi = std::max(hi, st.data_base() + t.end);
                }
                file_->drop_cache(lo, hi - lo);
            }
            host_.push_back(std::move(host));
            blocks_.push_back(b);
            F_ = b.imlp1.n;
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (Mage-Flow DiT): no transformer blocks");
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        t16_ = alloc<uint16_t>(allocs_, 256 + 2 * C_);
        temb_ = alloc<float>(allocs_, C_);
        mod_i_ = alloc<float>(allocs_, 6 * C_);
        mod_t_ = alloc<float>(allocs_, 6 * C_);
        nmod_ = alloc<float>(allocs_, 2 * C_);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void MfDit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                    const float* bias) {
    if (T <= 0) return;
    const auto* d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    if (w.type == kTypeF16) {
        gemm_.f16(x16, reinterpret_cast<const uint16_t*>(d), y, T, w.n, w.k);
    } else {
        gemm_.native(x16, w.type, d, y, T, w.n, w.k);
    }
    if (rs != nullptr || bias != nullptr) rows_back(y, rs, bias, T, w.n, s_);
}

uint64_t MfDit::activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows) const {
    const int64_t R = n * hw, L = n * max_text, X = std::max(R, L), TP = qi_attn_rows(hw + max_text);
    const int64_t band = std::max(L, mlp_rows > 0 ? std::min(R, mlp_rows) : R);
    return (uint64_t) (R + L) * C_ * 4 +                 // the two streams
           (uint64_t) X * C_ * 4 * 4 +                   // q, k, v, a
           (uint64_t) band * F_ * 4 +                    // the MLP's hidden rows
           (uint64_t) std::max(X * C_, band * F_) * 2 +  // the products' FP16 input
           (uint64_t) n * TP * C_ * 2 * 3 + (uint64_t) TP * C_ * 2 +   // the joint q, k, v (a context each), output
           (uint64_t) (R + L) * C_ * 2 +                 // the attention's outputs by stream
           (uint64_t) (hw + max_text) * hd_ * 4 +        // the rotary tables
           (uint64_t) TP * C_ * 4;                       // the plain attention's scratch
}

void MfDit::ensure(int64_t hw, int n, int64_t max_text) {
    if (hw <= cap_hw_ && n <= cap_n_ && max_text <= cap_text_ && (mlp_cap_ <= 0 || mlp_rows_ <= mlp_cap_)) return;
    drop_activations();
    const int64_t R = n * hw, L = n * max_text, X = std::max(R, L), TP = qi_attn_rows(hw + max_text);
    mlp_rows_ = std::max(L, mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R);
    img_ = alloc<float>(allocs_, R * C_);
    txt_ = alloc<float>(allocs_, L * C_);
    q_ = alloc<float>(allocs_, X * C_);
    k_ = alloc<float>(allocs_, X * C_);
    v_ = alloc<float>(allocs_, X * C_);
    a_ = alloc<float>(allocs_, X * C_);
    tq_ = alloc<float>(allocs_, L * C_);
    tk_ = alloc<float>(allocs_, L * C_);
    tv_ = alloc<float>(allocs_, L * C_);
    rs_ = alloc<float>(allocs_, std::max(X, mlp_rows_));
    rs_io_ = alloc<float>(allocs_, R);
    rs_to_ = alloc<float>(allocs_, L);
    hs_ = alloc<float>(allocs_, 2 * heads_);
    vpart_ = alloc<float>(allocs_, (hw + max_text) * heads_);
    gu_ = alloc<float>(allocs_, mlp_rows_ * F_);
    h16_ = alloc<uint16_t>(allocs_, std::max(X * C_, mlp_rows_ * F_));
    jq16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jk16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jv16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_);
    io16_ = alloc<uint16_t>(allocs_, R * C_);
    to16_ = alloc<uint16_t>(allocs_, L * C_);
    cos_ = alloc<float>(allocs_, hw * hd_ / 2);
    sin_ = alloc<float>(allocs_, hw * hd_ / 2);
    {
        std::vector<float> one((size_t) (max_text * hd_ / 2), 1.f), zero((size_t) (max_text * hd_ / 2), 0.f);
        cos_id_ = alloc<float>(allocs_, max_text * hd_ / 2);
        sin_id_ = alloc<float>(allocs_, max_text * hd_ / 2);
        gpu::copy(cos_id_, one.data(), one.size() * 4);
        gpu::copy(sin_id_, zero.data(), zero.size() * 4);
    }
    key_lim_ = alloc<int32_t>(allocs_, 2 * max_text);
    if (std::strcmp(qi_attention_path(), "plain (FP32)") == 0) attn_scratch_ = alloc<float>(allocs_, TP * C_);
    b0_img_ = alloc<float>(allocs_, hw * C_);
    b0_txt_ = alloc<float>(allocs_, max_text * C_);
    cap_hw_ = hw;
    cap_n_ = n;
    cap_text_ = max_text;
    dirty_ = 0;
    rope_h_ = rope_w_ = -1;
}

void MfDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) tq_, (void*) tk_, (void*) tv_, (void*) rs_, (void*) rs_io_, (void*) rs_to_, (void*) hs_,
                    (void*) vpart_})
        release(allocs_, p);
    tq_ = tk_ = tv_ = rs_ = rs_io_ = rs_to_ = hs_ = vpart_ = nullptr;
    for (void* p : {(void*) img_, (void*) txt_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) gu_,
                    (void*) h16_, (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) io16_,
                    (void*) to16_, (void*) cos_, (void*) sin_, (void*) cos_id_, (void*) sin_id_, (void*) key_lim_,
                    (void*) attn_scratch_, (void*) b0_img_, (void*) b0_txt_})
        release(allocs_, p);
    img_ = txt_ = q_ = k_ = v_ = a_ = gu_ = cos_ = sin_ = cos_id_ = sin_id_ = nullptr;
    attn_scratch_ = b0_img_ = b0_txt_ = nullptr;
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = io16_ = to16_ = nullptr;
    key_lim_ = nullptr;
    cap_hw_ = cap_n_ = cap_text_ = 0;
}

// the image rows' 2D multi-scale RoPE: pairs 0 .. 7 the frame (0 for a generated image), then axes_[1] / 2 pairs by
// the row and axes_[2] / 2 by the column, each centred on the image (rows -(h - h / 2) .. h / 2 - 1)
void MfDit::rope_tables(int64_t h, int64_t w) {
    if (h == rope_h_ && w == rope_w_) return;
    const int64_t half = hd_ / 2, a0 = axes_[0] / 2, a1 = axes_[1] / 2;
    std::vector<float> c((size_t) (h * w * half)), s(c.size());
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x)
            for (int64_t j = 0; j < half; ++j) {
                double pos = 0, dim = (double) axes_[0], k = (double) j;
                if (j >= a0 && j < a0 + a1) {
                    pos = (double) (y - (h - h / 2));
                    dim = (double) axes_[1];
                    k = (double) (j - a0);
                } else if (j >= a0 + a1) {
                    pos = (double) (x - (w - w / 2));
                    dim = (double) axes_[2];
                    k = (double) (j - a0 - a1);
                }
                const double ang = pos * std::pow(10000.0, -2.0 * k / dim);
                c[(size_t) ((y * w + x) * half + j)] = (float) std::cos(ang);
                s[(size_t) ((y * w + x) * half + j)] = (float) std::sin(ang);
            }
    gpu::stream_sync(s_);
    gpu::copy(cos_, c.data(), c.size() * 4);
    gpu::copy(sin_, s.data(), s.size() * 4);
    rope_h_ = h;
    rope_w_ = w;
}

// the time's embedding as the reference makes it in bf16 (it was trained with this rounding): the sinusoid of the
// bf16 sigma by bf16 frequencies (flipped: cos then sin), rounded to bf16, through the two linear layers; then temb's
// SiLU as FP16 (the modulations' input) at t16_ + 256 + C
void MfDit::time_embedding(float sigma) {
    const float t = round_bf16(sigma);
    std::vector<uint16_t> e(256);
    for (int i = 0; i < 128; ++i) {
        const float f = round_bf16(std::exp(-std::log(10000.f) * (float) i / 128.f));
        const float a = 1000.f * (t * f);
        e[(size_t) i] = f16_from_f32(round_bf16(std::cos(a)));
        e[(size_t) (128 + i)] = f16_from_f32(round_bf16(std::sin(a)));
    }
    gpu::copy_async(t16_, e.data(), e.size() * 2, s_);
    gpu::stream_sync(s_);   // e leaves scope
    product(t16_, nullptr, t1_, nullptr, temb_, 1, b_t1_);
    qi_silu(temb_, C_, t16_ + 256, s_);
    product(t16_ + 256, nullptr, t2_, nullptr, temb_, 1, b_t2_);
    gpu::copy_async(nmod_, temb_, (size_t) C_ * 4, s_);   // temb kept; its SiLU below
    qi_silu(nmod_, C_, t16_ + 256 + C_, s_);
}

void MfDit::set_context(int i, const float* states, int64_t rows) {
    Ctx& c = ctx_[i];
    gpu::stream_sync(s_);
    if (c.txt0 != nullptr) release(allocs_, c.txt0);
    c.txt0 = nullptr;
    c.rows = rows;
    if (rows <= 0) return;
    float* x = alloc<float>(allocs_, rows * txt_dim_);
    uint16_t* x16 = alloc<uint16_t>(allocs_, rows * txt_dim_);
    gpu::copy(x, states, (size_t) (rows * txt_dim_) * 4);
    qi_rms_norm16(x, txt_norm_, rows, txt_dim_, x16, s_);
    c.txt0 = alloc<float>(allocs_, rows * C_);
    product(x16, nullptr, txt_in_, nullptr, c.txt0, rows, b_txt_in_);
    gpu::stream_sync(s_);
    release(allocs_, x);
    release(allocs_, x16);
}

void MfDit::forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out) {
    const int64_t HW = h * w, C = C_, R = n * HW;
    int64_t L = 0, maxL = 0;
    for (int i = 0; i < n; ++i) {
        L += ctx_[i].rows;
        maxL = std::max(maxL, ctx_[i].rows);
    }
    ensure(HW, n, std::max(maxL, cap_text_));
    rope_tables(h, w);
    const int64_t TP = qi_attn_rows(HW + cap_text_);
    // the image's input projection (each context the same rows), the text stream's start
    qi_to_f16(x, HW * in_ch_, h16_, s_);
    product(h16_, nullptr, img_in_, nullptr, img_, HW, b_img_in_);
    for (int i = 1; i < n; ++i) gpu::copy_async(img_ + i * HW * C, img_, (size_t) (HW * C) * 4, s_);
    std::vector<int64_t> toff(2, 0);
    for (int i = 0; i < n; ++i) {
        toff[(size_t) i] = i == 0 ? 0 : toff[(size_t) i - 1] + ctx_[i - 1].rows;
        gpu::copy_async(txt_ + toff[(size_t) i] * C, ctx_[i].txt0, (size_t) (ctx_[i].rows * C) * 4, s_);
        fill_i32(key_lim_ + i * cap_text_, std::max<int64_t>(ctx_[i].rows, 1), (int32_t) (ctx_[i].rows + HW), s_);
    }
    // the joint rows past a context's end zero (the attention reads them up to T rounded up)
    for (int i = 0; i < n; ++i) {
        const int64_t T = ctx_[i].rows + HW;
        if (dirty_ > T)
            for (uint16_t* b : {jq16_, jk16_, jv16_})
                gpu::memset_async(b + (i * TP + T) * C, 0, (size_t) ((dirty_ - T) * C) * 2, s_);
    }
    for (int i = 0; i < n; ++i) dirty_ = std::max(dirty_, ctx_[i].rows + HW);
    time_embedding(sigma);
    const uint16_t* tsil = t16_ + 256 + C;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);
        auto f = [base](uint64_t off) { return reinterpret_cast<const float*>(base + off); };
        product(tsil, nullptr, b.imod, base, mod_i_, 1, f(b.b_imod));
        product(tsil, nullptr, b.tmod, base, mod_t_, 1, f(b.b_tmod));
        // q, k, v of the image rows and of the text rows
        rows16(LnMod{img_, mod_i_, mod_i_ + C, C}, R, C, h16_, rs_, s_);
        product(h16_, rs_, b.q, base, q_, R, f(b.b_q));
        product(h16_, rs_, b.k, base, k_, R, f(b.b_k));
        product(h16_, rs_, b.v, base, v_, R, f(b.b_v));
        rows16(LnMod{txt_, mod_t_, mod_t_ + C, C}, L, C, h16_, rs_, s_);
        product(h16_, rs_, b.tq, base, tq_, L, f(b.b_tq));
        product(h16_, rs_, b.tk, base, tk_, L, f(b.b_tk));
        product(h16_, rs_, b.tv, base, tv_, L, f(b.b_tv));
        // each context: its v heads' scales, its rows normed (the image's rotated) into [text, image], the attention,
        // its output back to each stream's rows (FP16, scaled by row)
        for (int i = 0; i < n; ++i) {
            const int64_t o = i * TP * C, Lt = ctx_[i].rows, t0 = toff[(size_t) i] * C;
            const float* vi = v_ + i * HW * C;
            head_max(tv_ + t0, Lt, vi, HW, heads_, hd_, vpart_, s_);
            head_scale(vpart_, Lt + HW, heads_, hs_ + i * heads_, s_);
            qk_rope_v16(tq_ + t0, tk_ + t0, tv_ + t0, f(b.ntq), f(b.ntk), nullptr, nullptr, false, hs_ + i * heads_, Lt,
                        heads_, hd_, jq16_ + o, jk16_ + o, jv16_ + o, s_);
            qk_rope_v16(q_ + i * HW * C, k_ + i * HW * C, vi, f(b.nq), f(b.nk), cos_, sin_, true, hs_ + i * heads_, HW,
                        heads_, hd_, jq16_ + o + Lt * C, jk16_ + o + Lt * C, jv16_ + o + Lt * C, s_);
            qi_attention(jq16_ + o, jk16_ + o, jv16_ + o, Lt + HW, heads_, hd_, Lt, jo16_, attn_scratch_, s_,
                         key_lim_ + i * cap_text_);
            rows16(Heads{jo16_, hs_ + i * heads_, C, hd_}, Lt, C, to16_ + t0, rs_to_ + toff[(size_t) i], s_);
            rows16(Heads{jo16_ + Lt * C, hs_ + i * heads_, C, hd_}, HW, C, io16_ + i * HW * C, rs_io_ + i * HW, s_);
        }
        product(io16_, rs_io_, b.o, base, a_, R, f(b.b_o));
        gate_add(img_, a_, mod_i_ + 2 * C, R, C, s_);
        product(to16_, rs_to_, b.to, base, a_, L, f(b.b_to));
        gate_add(txt_, a_, mod_t_ + 2 * C, L, C, s_);
        // the MLPs, the image's a band of rows at a time
        for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
            const int64_t mr = std::min(mlp_rows_, R - m0);
            rows16(LnMod{img_ + m0 * C, mod_i_ + 3 * C, mod_i_ + 4 * C, C}, mr, C, h16_, rs_, s_);
            product(h16_, rs_, b.imlp1, base, gu_, mr, f(b.b_imlp1));
            rows16(Gelu{gu_, F_}, mr, F_, h16_, rs_, s_);
            product(h16_, rs_, b.imlp2, base, a_, mr, f(b.b_imlp2));
            gate_add(img_ + m0 * C, a_, mod_i_ + 5 * C, mr, C, s_);
        }
        rows16(LnMod{txt_, mod_t_ + 3 * C, mod_t_ + 4 * C, C}, L, C, h16_, rs_, s_);
        product(h16_, rs_, b.tmlp1, base, gu_, L, f(b.b_tmlp1));
        rows16(Gelu{gu_, F_}, L, F_, h16_, rs_, s_);
        product(h16_, rs_, b.tmlp2, base, a_, L, f(b.b_tmlp2));
        gate_add(txt_, a_, mod_t_ + 5 * C, L, C, s_);
        blocks_stream_.end(bi);
        if (bi == 0 && keep0_) {
            gpu::copy_async(b0_img_, img_, (size_t) (HW * C) * 4, s_);
            gpu::copy_async(b0_txt_, txt_, (size_t) (ctx_[0].rows * C) * 4, s_);
        }
    }
    // the output: AdaLayerNormContinuous (its emb: scale, then shift) and the projection to the latent's channels
    product(tsil, nullptr, nout_, nullptr, nmod_, 1, b_nout_);
    rows16(LnMod{img_, nmod_ + C, nmod_, C}, R, C, h16_, rs_, s_);
    product(h16_, rs_, pout_, nullptr, out, R, b_pout_);
}

void MfDit::sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                   const std::function<bool(int, int)>& step) {
    const int n = cfg > 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const int64_t HW = h * w;
    float* out = alloc<float>(allocs_, n * HW * in_ch_);
    const int steps = (int) sigmas.size() - 1;
    for (int i = 0; i < steps; ++i) {
        forward(x, sigmas[(size_t) i], h, w, n, out);
        euler(x, out, HW * in_ch_, n, cfg, sigmas[(size_t) i + 1] - sigmas[(size_t) i], s_);
        if (step && !step(i + 1, steps)) break;
    }
    gpu::stream_sync(s_);
    release(allocs_, out);
}

}  // namespace strata::image
